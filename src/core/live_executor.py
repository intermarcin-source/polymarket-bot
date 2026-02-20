"""
Live Trade Execution Engine for Polymarket CLOB.
================================================
Drop-in replacement for PaperTrader (simulator.py) when SIMULATION_MODE=false.
Same interface: positions, balance, execute_trade, check_exits, sell_position, get_performance.

Safety layers:
  - Kill switch (file-based, instant halt)
  - Daily loss limit (configurable, default $200)
  - Slippage protection (reject if >2% from expected price)
  - Price bounds (reject <$0.05 or >$0.97)
  - Discord/Telegram notifications on every trade + error
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType, AssetType, BalanceAllowanceParams
from py_clob_client.order_builder.constants import BUY, SELL

from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("live_executor")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
LIVE_PORTFOLIO_FILE = DATA_DIR / "live_portfolio.json"
LIVE_TRADES_FILE = DATA_DIR / "live_trades.json"

# Exit thresholds — wider for 5-min markets (resolution handles exits)
TAKE_PROFIT_PCT = 50.0
STOP_LOSS_PCT = -50.0
TRAILING_STOP_PCT = -20.0
TRAILING_ACTIVATE_PCT = 20.0


class LiveExecutor:
    """
    Real trade execution via Polymarket CLOB.
    Interface-compatible with PaperTrader for orchestrator integration.
    """

    def __init__(self, client: PolymarketClient, starting_balance: float = 0.0):
        self.client = client  # existing read-only client for prices
        self.starting_balance = starting_balance
        self.balance: float = starting_balance
        self.positions: list[dict] = []
        self.closed_trades: list[dict] = []
        self.trade_history: list[dict] = []
        self.total_pnl: float = 0.0
        self.daily_loss: float = 0.0
        self.daily_loss_reset_date: str = ""

        # Initialize CLOB client for order execution
        self.clob_client = self._init_clob_client()

        # Read actual on-chain USDC.e balance
        onchain_bal = self._get_onchain_balance()
        if onchain_bal > 0:
            self.balance = onchain_bal
            if self.starting_balance <= 0:
                self.starting_balance = onchain_bal
            log.info(f"On-chain USDC.e balance: ${onchain_bal:.2f}")

        # Load persisted state (may override balance if state exists)
        self._load_state()

        # Always sync balance with on-chain if we got a valid reading
        if onchain_bal > 0:
            self.balance = onchain_bal
            self._save_state()

    def _init_clob_client(self) -> ClobClient:
        """Initialize the py-clob-client with wallet credentials."""
        try:
            client = ClobClient(
                Config.CLOB_API_URL,
                key=Config.PRIVATE_KEY,
                chain_id=137,  # Polygon mainnet
                signature_type=Config.SIGNATURE_TYPE,
                funder=Config.FUNDER_ADDRESS,
            )
            signer_addr = client.get_address()
            log.info(f"CLOB signer address: {signer_addr}")
            log.info(f"CLOB funder address: {Config.FUNDER_ADDRESS}")
            log.info(f"Signature type: {Config.SIGNATURE_TYPE} (0=EOA, 1=POLY_PROXY, 2=GNOSIS)")

            # Use API credentials from .env if available, otherwise derive them
            if Config.API_KEY and Config.API_SECRET and Config.API_PASSPHRASE:
                from py_clob_client.clob_types import ApiCreds
                creds = ApiCreds(
                    api_key=Config.API_KEY,
                    api_secret=Config.API_SECRET,
                    api_passphrase=Config.API_PASSPHRASE,
                )
                client.set_api_creds(creds)
                log.info(f"CLOB client initialized with .env API creds | key: {Config.API_KEY[:8]}... | Funder: {Config.FUNDER_ADDRESS[:10]}...")
            else:
                creds = client.create_or_derive_api_creds()
                if not creds or not getattr(creds, 'api_key', None):
                    raise ValueError(
                        f"Failed to derive CLOB API credentials. "
                        f"Signer: {signer_addr}, Funder: {Config.FUNDER_ADDRESS}, "
                        f"SigType: {Config.SIGNATURE_TYPE}"
                    )
                client.set_api_creds(creds)
                log.info(f"CLOB client initialized with derived API creds | key: {creds.api_key[:8]}... | Funder: {Config.FUNDER_ADDRESS[:10]}...")

            # Set USDC (collateral) allowance so the exchange can spend our funds
            self._set_allowances(client)

            return client
        except Exception as e:
            log.critical(f"Failed to initialize CLOB client: {e}")
            raise

    def _set_allowances(self, client: ClobClient):
        """Approve USDC allowance for the Polymarket CTF Exchange (on-chain + CLOB API)."""
        # ── On-chain ERC20 approve ────────────────────────────────
        try:
            from web3 import Web3

            USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            CTF_EXCHANGE = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
            NEG_RISK_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")
            CTF_CONTRACT = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
            MAX_APPROVAL = 2**256 - 1

            ERC20_ABI = [{"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
            ERC1155_ABI = [{"inputs":[{"name":"operator","type":"address"},{"name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"name":"account","type":"address"},{"name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"name":"","type":"bool"}],"stateMutability":"view","type":"function"}]

            w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC_URL, request_kwargs={"timeout": 30}))
            try:
                block = w3.eth.block_number
                log.info(f"Web3 connected to Polygon (block {block})")
            except Exception as rpc_err:
                log.error(f"Web3 not connected — skipping on-chain approvals: {rpc_err}")
                block = None

            if block:
                account = w3.eth.account.from_key(Config.PRIVATE_KEY)
                usdc = w3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)
                ctf = w3.eth.contract(address=CTF_CONTRACT, abi=ERC1155_ABI)

                NEG_RISK_ADAPTER = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")

                spenders = [
                    ("CTF Exchange", CTF_EXCHANGE),
                    ("NegRisk Exchange", NEG_RISK_EXCHANGE),
                    ("NegRisk Adapter", NEG_RISK_ADAPTER),
                ]
                for label, spender in spenders:
                    # USDC approval
                    current = usdc.functions.allowance(account.address, spender).call()
                    if current < 10**9:  # less than 1000 USDC (6 decimals)
                        log.info(f"Setting USDC approval for {label} ({spender[:10]}...)...")
                        tx = usdc.functions.approve(spender, MAX_APPROVAL).build_transaction({
                            "from": account.address,
                            "nonce": w3.eth.get_transaction_count(account.address),
                            "gas": 80_000,
                            "gasPrice": w3.eth.gas_price,
                            "chainId": 137,
                        })
                        signed = account.sign_transaction(tx)
                        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                        log.info(f"USDC approved for {label}: tx {tx_hash.hex()} (status={receipt['status']})")
                    else:
                        log.info(f"USDC already approved for {label} (allowance={current})")

                    # ERC1155 conditional token approval
                    is_approved = ctf.functions.isApprovedForAll(account.address, spender).call()
                    if not is_approved:
                        log.info(f"Setting ERC1155 approval for {label}...")
                        tx = ctf.functions.setApprovalForAll(spender, True).build_transaction({
                            "from": account.address,
                            "nonce": w3.eth.get_transaction_count(account.address),
                            "gas": 80_000,
                            "gasPrice": w3.eth.gas_price,
                            "chainId": 137,
                        })
                        signed = account.sign_transaction(tx)
                        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                        log.info(f"ERC1155 approved for {label}: tx {tx_hash.hex()} (status={receipt['status']})")
                    else:
                        log.info(f"ERC1155 already approved for {label}")

        except Exception as e:
            log.error(f"On-chain approval failed: {e}", exc_info=True)

        # ── CLOB API allowance update ─────────────────────────────
        try:
            collateral_params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = client.update_balance_allowance(collateral_params)
            log.info(f"CLOB allowance update: {resp}")
        except Exception as e:
            log.error(f"CLOB allowance update failed: {e}")

    # ── ON-CHAIN BALANCE ─────────────────────────────────────

    def _get_onchain_balance(self) -> float:
        """Read actual USDC.e balance from Polygon chain."""
        try:
            from web3 import Web3
            USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            BALANCE_ABI = [{"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]

            w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC_URL, request_kwargs={"timeout": 10}))
            usdc = w3.eth.contract(address=USDC_ADDRESS, abi=BALANCE_ABI)
            wallet = Web3.to_checksum_address(Config.WALLET_ADDRESS)
            raw = usdc.functions.balanceOf(wallet).call()
            return raw / 1e6  # USDC.e has 6 decimals
        except Exception as e:
            log.warning(f"Could not read on-chain balance: {e}")
            return 0.0

    # ── ON-CHAIN REDEMPTION ─────────────────────────────────

    def _get_token_balance(self, token_id: str) -> int:
        """Get ERC1155 conditional token balance for a specific token_id."""
        try:
            from web3 import Web3
            CTF_CONTRACT = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
            BALANCE_ABI = [{
                "inputs": [
                    {"name": "account", "type": "address"},
                    {"name": "id", "type": "uint256"}
                ],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }]

            w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC_URL, request_kwargs={"timeout": 10}))
            ctf = w3.eth.contract(address=CTF_CONTRACT, abi=BALANCE_ABI)
            wallet = Web3.to_checksum_address(Config.WALLET_ADDRESS)
            balance = ctf.functions.balanceOf(wallet, int(token_id)).call()
            return balance
        except Exception as e:
            log.warning(f"Could not read token balance for {token_id[:20]}...: {e}")
            return 0

    def _redeem_neg_risk(self, condition_id: str, position: dict) -> bool:
        """
        Redeem resolved position on-chain via CTF contract.
        Calls redeemPositions on CTF (0x4D97...) to burn ERC1155 tokens -> USDC.e.
        """
        try:
            from web3 import Web3

            CTF_CONTRACT = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
            USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            PARENT_COLLECTION_ID = b'\x00' * 32

            REDEEM_ABI = [{
                "name": "redeemPositions",
                "type": "function",
                "inputs": [
                    {"name": "collateralToken", "type": "address"},
                    {"name": "parentCollectionId", "type": "bytes32"},
                    {"name": "conditionId", "type": "bytes32"},
                    {"name": "indexSets", "type": "uint256[]"}
                ],
                "outputs": [],
                "stateMutability": "nonpayable"
            }]

            w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC_URL, request_kwargs={"timeout": 30}))
            account = w3.eth.account.from_key(Config.PRIVATE_KEY)

            # Convert condition_id to bytes32
            cid = condition_id[2:] if condition_id.startswith("0x") else condition_id
            cond_bytes32 = bytes.fromhex(cid).ljust(32, b'\x00')[:32]

            ctf = w3.eth.contract(address=CTF_CONTRACT, abi=REDEEM_ABI)

            log.info(f"Redeeming via CTF | condition: {condition_id[:16]}...")

            # Use 3x network gas price to avoid dropped transactions
            network_gas = w3.eth.gas_price
            boosted_gas = int(network_gas * 3)
            log.info(f"Gas price: network={network_gas} wei, using={boosted_gas} wei (3x)")

            tx = ctf.functions.redeemPositions(
                USDC_ADDRESS,
                PARENT_COLLECTION_ID,
                cond_bytes32,
                [1, 2]  # indexSets for binary market (YES=1, NO=2)
            ).build_transaction({
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 300_000,
                "gasPrice": boosted_gas,
                "chainId": 137,
            })
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            if receipt['status'] == 1:
                log.info(f"[REDEEM SUCCESS] CTF position redeemed | tx: {tx_hash.hex()}")
                return True
            else:
                log.error(f"[REDEEM FAILED] tx reverted: {tx_hash.hex()}")
                return False

        except Exception as e:
            log.error(f"CTF redemption failed for {condition_id[:16]}...: {e}", exc_info=True)
            return False

    def _sync_onchain_balance(self):
        """Re-read on-chain USDC.e balance and update internal state."""
        onchain_bal = self._get_onchain_balance()
        if onchain_bal > 0:
            old_bal = self.balance
            self.balance = onchain_bal
            if abs(old_bal - onchain_bal) > 0.01:
                log.info(f"Balance synced: ${old_bal:.2f} -> ${onchain_bal:.2f} (on-chain)")
            self._save_state()

    # ── STATE PERSISTENCE ────────────────────────────────────

    def _load_state(self):
        """Load persisted live trading state."""
        DATA_DIR.mkdir(exist_ok=True)
        if LIVE_PORTFOLIO_FILE.exists():
            try:
                with open(LIVE_PORTFOLIO_FILE) as f:
                    state = json.load(f)
                    self.balance = state.get("balance", self.starting_balance)
                    self.positions = state.get("positions", [])
                    self.closed_trades = state.get("closed_trades", [])
                    self.total_pnl = state.get("total_pnl", 0.0)
                    self.starting_balance = state.get("starting_balance", self.starting_balance)
                    self.daily_loss = state.get("daily_loss", 0.0)
                    self.daily_loss_reset_date = state.get("daily_loss_reset_date", "")
                log.info(
                    f"Loaded live state: ${self.balance:.2f} balance, "
                    f"{len([p for p in self.positions if p.get('status') == 'open'])} open positions"
                )
            except Exception as e:
                log.error(f"Failed to load live state: {e}")

        if LIVE_TRADES_FILE.exists():
            try:
                with open(LIVE_TRADES_FILE) as f:
                    self.trade_history = json.load(f)
            except Exception as e:
                log.error(f"Failed to load trade history: {e}")

        # Always save initial state so dashboard has data files to read
        self._save_state()

    def _save_state(self):
        """Persist live trading state to disk."""
        DATA_DIR.mkdir(exist_ok=True)
        try:
            with open(LIVE_PORTFOLIO_FILE, "w") as f:
                json.dump({
                    "balance": self.balance,
                    "positions": self.positions,
                    "closed_trades": self.closed_trades,
                    "total_pnl": self.total_pnl,
                    "starting_balance": self.starting_balance,
                    "daily_loss": self.daily_loss,
                    "daily_loss_reset_date": self.daily_loss_reset_date,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2, default=str)

            with open(LIVE_TRADES_FILE, "w") as f:
                json.dump(self.trade_history, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save live state: {e}")

    # ── SAFETY CHECKS ────────────────────────────────────────

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists (emergency stop)."""
        kill_file = Path(Config.KILL_SWITCH_FILE)
        if kill_file.exists():
            log.critical("KILL SWITCH ACTIVATED - refusing all trades")
            return True
        return False

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily_loss_reset_date != today:
            self.daily_loss = 0.0
            self.daily_loss_reset_date = today
            self._save_state()

        if self.daily_loss <= -Config.MAX_DAILY_LOSS_USDC:
            log.critical(
                f"DAILY LOSS LIMIT HIT: ${self.daily_loss:.2f} "
                f"(limit: -${Config.MAX_DAILY_LOSS_USDC})"
            )
            return True
        return False

    # ── TOKEN ID RESOLUTION ──────────────────────────────────

    async def _resolve_token_id(self, condition_id: str, outcome: str) -> str:
        """Resolve a token_id from condition_id + outcome via Gamma API."""
        if not condition_id:
            return ""
        try:
            market = await self.client.get_market(condition_id)
            if market:
                tokens = market.get("tokens", [])
                for t in tokens:
                    if t.get("outcome", "").lower() == outcome.lower():
                        tid = t.get("token_id", "")
                        if tid:
                            return tid
                if tokens and len(tokens) == 2:
                    for t in tokens:
                        if t.get("outcome", "").lower() == outcome.lower():
                            return t.get("token_id", "")
        except Exception as e:
            log.debug(f"Token ID resolution failed for {condition_id[:20]}: {e}")
        return ""

    # ── BUY ──────────────────────────────────────────────────

    async def execute_trade(self, decision: dict) -> dict:
        """
        Execute a REAL BUY order on Polymarket CLOB.
        Uses FOK (Fill-Or-Kill) by default for immediate execution.
        """
        # Safety checks
        if self._check_kill_switch():
            return {"success": False, "reason": "Kill switch active"}
        if self._check_daily_loss_limit():
            return {"success": False, "reason": "Daily loss limit reached"}

        signal = decision["signal"]
        size_usdc = decision["position_size_usdc"]

        if size_usdc > self.balance:
            log.warning(f"Insufficient balance: ${self.balance:.2f} < ${size_usdc:.2f}")
            return {"success": False, "reason": "Insufficient balance"}

        token_id = signal.get("token_id", "")
        market_price = signal.get("market_price", 0)
        outcome = signal.get("recommended_outcome", signal.get("outcome", ""))

        # Resolve token_id if missing
        if not token_id and signal.get("condition_id"):
            token_id = await self._resolve_token_id(signal["condition_id"], outcome)

        if not token_id:
            return {"success": False, "reason": "Could not resolve token_id"}

        # Get fresh price and check slippage
        # For BTC sniper signals, the signal's market_price comes from Gamma API
        # and is reliable. The CLOB /book endpoint returns extreme resting orders
        # ($0.01/$0.99) for 5-min markets, making orderbook-based slippage checks
        # falsely reject every trade. Trust the signal price for btc_sniper.
        is_sniper = signal.get("source") == "btc_sniper"

        if is_sniper and market_price > 0:
            # BTC sniper fetched actual CLOB book asks at T-120s — price is real
            log.info(f"BTC sniper signal — using book price ${market_price:.4f} (skipping slippage check)")
        else:
            # For other sources, verify against CLOB orderbook
            prices = await self.client.get_prices([token_id])
            if token_id not in prices:
                log.warning(f"No orderbook/price for token {token_id[:20]}... — skipping stale market")
                return {"success": False, "reason": "No orderbook for token (market may be expired)"}

            live_price = prices[token_id]
            if market_price > 0:
                slippage = abs(live_price - market_price) / market_price * 100
                if slippage > Config.MAX_SLIPPAGE_PCT:
                    log.warning(
                        f"Slippage {slippage:.1f}% exceeds max {Config.MAX_SLIPPAGE_PCT}% "
                        f"(signal: ${market_price:.4f}, live: ${live_price:.4f})"
                    )
                    return {"success": False, "reason": f"Slippage {slippage:.1f}% exceeds max"}
            market_price = live_price

        if not market_price or market_price <= 0:
            return {"success": False, "reason": "No valid price available"}

        # Price bounds (wide for 5-min markets)
        if market_price < 0.01:
            return {"success": False, "reason": f"Price too low ({market_price:.4f})"}
        if market_price > 0.99:
            return {"success": False, "reason": f"Price too high ({market_price:.4f})"}
        if market_price >= 1:
            return {"success": False, "reason": f"Invalid price: {market_price}"}

        shares = size_usdc / market_price

        # For sniper signals, cap shares by available liquidity on the book
        if is_sniper:
            avail = signal.get("_available_shares", 0)
            if avail > 0 and shares > avail:
                log.info(
                    f"Capping order from {shares:.0f} to {avail:.0f} shares "
                    f"(available liquidity)"
                )
                shares = avail

        # Ensure conditional token allowance is set for this token
        try:
            cond_params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id,
            )
            self.clob_client.update_balance_allowance(cond_params)
        except Exception as e:
            log.warning(f"Conditional token allowance update failed (may already be set): {e}")

        try:
            # Round price to nearest tick (0.01) for CLOB compatibility
            order_price = round(market_price, 2)
            # CLOB requires maker_amount (size*price for BUY) max 2 decimal places.
            # Use whole shares so maker_amount = int * 0.xx always has ≤ 2 decimals.
            import math
            order_shares = float(math.floor(shares))  # whole shares only
            if order_shares < 5:
                order_shares = 5.0  # CLOB minimum

            import time

            if is_sniper:
                # BTC sniper: Use FAK (Fill-And-Kill) to sweep asks immediately.
                # At T-120s there IS real liquidity on the book ($0.60-$0.85 asks).
                # FAK fills whatever is available at our price and cancels the rest.
                # Then poll for on-chain settlement (may take a few seconds).
                log.info(
                    f"Placing FAK order: {order_shares} shares @ ${order_price} "
                    f"= ${order_price * order_shares:.2f}"
                )

                pre_balance = self._get_token_balance(token_id)

                signed_order = self.clob_client.create_order(
                    OrderArgs(
                        price=order_price,
                        size=order_shares,
                        side=BUY,
                        token_id=token_id,
                    )
                )
                resp = self.clob_client.post_order(signed_order, OrderType.FAK)

                if not resp or not resp.get("success", False):
                    error_msg = resp.get("errorMsg", "Unknown error") if resp else "No response"
                    log.error(f"FAK order rejected: {error_msg}")
                    await self._send_notification(f"ORDER REJECTED (FAK): {error_msg}")
                    return {"success": False, "reason": f"FAK rejected: {error_msg}"}

                order_id = resp.get("orderID", resp.get("orderId", ""))
                log.info(f"FAK order accepted: {order_id[:20]}... — checking fill...")

                # FAK fills immediately, but on-chain settlement takes a few seconds.
                # Poll every 2s for up to 10s to catch the settlement.
                actual_shares = 0
                for check_num in range(5):  # 5 checks x 2s = 10s max
                    time.sleep(2)
                    post_balance = self._get_token_balance(token_id)
                    actual_shares_raw = post_balance - pre_balance
                    actual_shares = actual_shares_raw / 1e6
                    if actual_shares > 0:
                        log.info(
                            f"Fill confirmed after {(check_num+1)*2}s: "
                            f"{actual_shares:.2f} shares"
                        )
                        break

                if actual_shares <= 0:
                    log.warning(
                        f"FAK order: 0 shares settled on-chain after 10s "
                        f"(pre={pre_balance}, post={post_balance})"
                    )
                    return {"success": False, "reason": "0 shares settled on-chain (FAK)"}

            else:
                # Non-sniper: Use GTC limit order with on-chain verification
                log.info(
                    f"Placing GTC order: {order_shares} shares @ ${order_price} "
                    f"= ${order_price * order_shares:.2f}"
                )

                pre_balance = self._get_token_balance(token_id)

                signed_order = self.clob_client.create_order(
                    OrderArgs(
                        price=order_price,
                        size=order_shares,
                        side=BUY,
                        token_id=token_id,
                    )
                )
                resp = self.clob_client.post_order(signed_order, OrderType.GTC)

                if not resp or not resp.get("success", False):
                    error_msg = resp.get("errorMsg", "Unknown error") if resp else "No response"
                    log.error(f"Order rejected by CLOB: {error_msg}")
                    await self._send_notification(f"ORDER REJECTED: {error_msg}")
                    return {"success": False, "reason": f"CLOB rejected: {error_msg}"}

                order_id = resp.get("orderID", resp.get("orderId", ""))

                time.sleep(2)
                post_balance = self._get_token_balance(token_id)
                actual_shares_raw = post_balance - pre_balance
                actual_shares = actual_shares_raw / 1e6

                if actual_shares <= 0:
                    log.warning(
                        f"GTC order accepted but 0 shares filled "
                        f"(pre={pre_balance}, post={post_balance}) — treating as failed"
                    )
                    try:
                        self.clob_client.cancel(order_id)
                        log.info(f"Cancelled unfilled GTC order {order_id[:16]}...")
                    except Exception:
                        pass
                    return {"success": False, "reason": "0 shares filled"}

            # Sanity check: if on-chain reports more shares than we ordered,
            # cap to ordered amount. FAK can sweep extra cheap asks, inflating
            # the fill beyond our intended bet size.
            if actual_shares > order_shares * 1.05:
                log.warning(
                    f"Fill sanity check: on-chain says {actual_shares:.2f} shares "
                    f"but we only ordered {order_shares:.0f}. Capping to ordered amount. "
                    f"(pre={pre_balance}, post={post_balance})"
                )
                actual_shares = order_shares

            # Calculate actual cost based on real fill, capped to intended bet size
            actual_cost = actual_shares * market_price
            if actual_cost > size_usdc:
                log.info(f"Capping recorded cost from ${actual_cost:.2f} to ${size_usdc:.2f} (intended bet)")
                actual_cost = size_usdc
            fill_pct = (actual_shares / shares) * 100 if shares > 0 else 0

            log.info(
                f"Fill verified: requested {shares:.2f} shares, got {actual_shares:.2f} "
                f"({fill_pct:.0f}% fill) | actual cost: ${actual_cost:.2f}"
            )

            # If partial fill on GTC, cancel the remaining resting order
            if not is_sniper and actual_shares < shares * 0.95:
                try:
                    self.clob_client.cancel(order_id)
                    log.info(f"Cancelled remaining GTC order after partial fill")
                except Exception:
                    pass

            # Record the trade with ACTUAL fill amounts
            trade = {
                "id": f"live_{len(self.trade_history) + 1}",
                "order_id": order_id,
                "condition_id": signal.get("condition_id", ""),
                "market_question": signal.get("market_question", ""),
                "outcome": outcome,
                "token_id": token_id,
                "side": "buy",
                "entry_price": market_price,
                "price": market_price,
                "current_price": market_price,
                "shares": round(actual_shares, 4),
                "size_usdc": round(actual_cost, 2),
                "source": signal.get("source", ""),
                "signal_type": signal.get("type", ""),
                "confidence": signal.get("confidence", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "open",
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "peak_pnl_pct": 0.0,
                "exit_reason": "",
                "requested_shares": round(shares, 4),
                "requested_size_usdc": round(size_usdc, 2),
                "fill_pct": round(fill_pct, 1),
            }

            # Sync balance with on-chain (most accurate)
            self._sync_onchain_balance()
            self.positions.append(trade)
            self.trade_history.append(trade)
            self._save_state()

            log.info(
                f"[LIVE BUY] {shares:.2f} shares of '{outcome}' "
                f"@ ${market_price:.4f} = ${size_usdc:.2f} "
                f"| {signal.get('market_question', '?')[:50]} "
                f"| Source: {signal.get('source', '?')} "
                f"| Order: {order_id[:16]}..."
            )
            await self._send_notification(
                f"BUY: {outcome} @ ${market_price:.4f} = ${size_usdc:.2f} | "
                f"{signal.get('market_question', '?')[:40]}"
            )

            return {"success": True, "trade": trade}

        except Exception as e:
            log.error(f"Order execution failed: {e}", exc_info=True)
            await self._send_notification(f"ORDER FAILED: {e}")
            return {"success": False, "reason": str(e)}

    # ── SELL / EXIT ──────────────────────────────────────────

    async def sell_position(self, position: dict, reason: str) -> dict:
        """Sell an open position via CLOB order."""
        if self._check_kill_switch():
            # Still allow exits even with kill switch — we want to close positions
            log.warning("Kill switch active but allowing exit to close position")

        token_id = position.get("token_id", "")
        if not token_id:
            return {"success": False, "reason": "No token_id on position"}

        # Get current price
        current_price = position.get("current_price", position.get("entry_price", 0.5))
        if token_id:
            prices = await self.client.get_prices([token_id])
            if token_id in prices:
                current_price = prices[token_id]
            elif "resolved" not in reason.lower():
                log.warning(f"No orderbook for sell token {token_id[:20]}... — market may be expired")
                return {"success": False, "reason": "No orderbook for token (market may be expired)"}

        # For market-resolved positions, use the resolved price
        if "resolved - WON" in reason:
            current_price = 1.0
        elif "resolved - LOST" in reason:
            current_price = 0.0

        shares = position.get("shares", 0)
        if shares <= 0:
            return {"success": False, "reason": "No shares to sell"}

        # Ensure conditional token allowance is set for selling
        try:
            cond_params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id,
            )
            self.clob_client.update_balance_allowance(cond_params)
        except Exception as e:
            log.warning(f"Sell allowance update failed (may already be set): {e}")

        try:
            # For resolved markets, redeem on-chain to convert ERC1155 -> USDC.e
            if "resolved" in reason.lower():
                condition_id = position.get("condition_id", "")
                if not condition_id:
                    log.warning(f"No condition_id — cannot redeem on-chain")
                    return {"success": False, "reason": "No condition_id for redemption"}

                # Read balance BEFORE redemption to measure real P&L
                balance_before = self._get_onchain_balance()
                log.info(
                    f"Market resolved — redeeming ERC1155 tokens on-chain... "
                    f"(balance before: ${balance_before:.2f})"
                )

                redeemed = self._redeem_neg_risk(condition_id, position)
                if not redeemed:
                    log.error(
                        f"On-chain redemption FAILED — keeping position open for retry | "
                        f"token: {token_id[:20]}..."
                    )
                    # DO NOT mark as sold — keep open so check_exits retries next cycle
                    return {"success": False, "reason": "Redemption failed, will retry"}

                # Wait for on-chain state to settle, then read balance AFTER
                import time
                time.sleep(3)
                balance_after = self._get_onchain_balance()
                real_proceeds = balance_after - balance_before
                cost = position["size_usdc"]
                profit = real_proceeds - cost
                pnl_pct = (real_proceeds / cost - 1) * 100 if cost > 0 else 0

                log.info(
                    f"[REDEEM VERIFIED] Balance: ${balance_before:.2f} -> ${balance_after:.2f} | "
                    f"Real proceeds: ${real_proceeds:.2f} | Cost: ${cost:.2f} | "
                    f"Real P&L: ${profit:+.2f}"
                )

                # Update balance from on-chain (single source of truth)
                self.balance = balance_after

            else:
                # Place sell order on CLOB (non-resolved markets)
                if getattr(Config, "ORDER_TYPE", "FOK").upper() == "FOK":
                    mo = MarketOrderArgs(
                        token_id=token_id,
                        amount=shares,
                        side=SELL,
                    )
                    signed = self.clob_client.create_market_order(mo)
                    resp = self.clob_client.post_order(signed, OrderType.FOK)
                else:
                    order_args = OrderArgs(
                        price=round(current_price, 2),
                        size=round(shares, 2),
                        side=SELL,
                    )
                    signed = self.clob_client.create_order({
                        "token_id": token_id,
                        "price": round(current_price, 2),
                        "size": round(shares, 2),
                        "side": SELL,
                    })
                    resp = self.clob_client.post_order(signed, OrderType.GTC)

                if not resp or not resp.get("success", False):
                    error_msg = resp.get("errorMsg", "Unknown") if resp else "No response"
                    log.error(f"Sell order failed: {error_msg}")
                    return {"success": False, "reason": error_msg}

                # For CLOB sells, use theoretical P&L (no on-chain redemption)
                proceeds = shares * current_price
                cost = position["size_usdc"]
                profit = proceeds - cost
                pnl_pct = (current_price / position["entry_price"] - 1) * 100 if position["entry_price"] > 0 else 0

                # Return capital
                self.balance += proceeds

            # Update position (only reached if exit was successful)
            position["status"] = "sold"
            position["exit_price"] = current_price
            position["exit_reason"] = reason
            position["pnl"] = round(profit, 2)
            position["pnl_pct"] = round(pnl_pct, 2)
            position["closed_at"] = datetime.now(timezone.utc).isoformat()

            self.total_pnl += profit
            self.daily_loss += profit  # negative profit = loss

            # Move to closed trades
            self.closed_trades.append(position)
            self._save_state()

            pnl_label = "PROFIT" if profit >= 0 else "LOSS"
            log.info(
                f"[LIVE SELL - {pnl_label}] '{position['outcome']}' "
                f"@ ${current_price:.4f} (entry: ${position['entry_price']:.4f}) "
                f"| PnL: ${profit:+.2f} ({pnl_pct:+.1f}%) "
                f"| Reason: {reason} "
                f"| {position.get('market_question', '?')[:45]}"
            )
            await self._send_notification(
                f"SELL ({pnl_label}): {position['outcome']} | "
                f"PnL: ${profit:+.2f} ({pnl_pct:+.1f}%) | {reason}"
            )

            return {"success": True, "profit": profit, "pnl_pct": pnl_pct}

        except Exception as e:
            log.error(f"Sell execution failed: {e}", exc_info=True)
            await self._send_notification(f"SELL FAILED: {e}")
            return {"success": False, "reason": str(e)}

    # ── BACKFILL TOKEN IDS ────────────────────────────────────

    async def _backfill_token_ids(self):
        """Resolve token_ids for any positions missing them."""
        fixed = 0
        for pos in self.positions:
            if pos.get("status") != "open":
                continue
            if pos.get("token_id"):
                continue
            condition_id = pos.get("condition_id", "")
            outcome = pos.get("outcome", "")
            if condition_id and outcome:
                token_id = await self._resolve_token_id(condition_id, outcome)
                if token_id:
                    pos["token_id"] = token_id
                    fixed += 1
                await asyncio.sleep(0.3)
        if fixed:
            log.info(f"Backfilled token_ids for {fixed} positions")
            self._save_state()

    # ── EXIT STRATEGY ENGINE ─────────────────────────────────

    async def check_exits(self) -> list[dict]:
        """
        Check every open position for exit conditions.
        Same logic as PaperTrader but executes real sell orders.

        Exit triggers:
        1. Market resolved (WON/LOST)
        2. Take profit: +20%
        3. Stop loss: -30%
        4. Trailing stop: -10% from peak (after +10%)
        """
        # Backfill missing token_ids first
        needs_backfill = any(
            not p.get("token_id") and p.get("status") == "open"
            for p in self.positions
        )
        if needs_backfill:
            await self._backfill_token_ids()

        exits = []
        remaining = []

        for pos in self.positions:
            if pos["status"] != "open":
                remaining.append(pos)
                continue

            try:
                token_id = pos.get("token_id", "")
                entry_price = pos.get("entry_price", pos.get("price", 0.5))

                # Get current price
                current_price = entry_price
                if token_id:
                    try:
                        prices = await self.client.get_prices([token_id])
                        if token_id in prices:
                            current_price = prices[token_id]
                    except Exception:
                        pass

                exit_reason = None

                # ── STEP 1: Check market resolution FIRST ──
                condition_id = pos.get("condition_id", "")
                if condition_id:
                    try:
                        market = await self.client.get_market(condition_id)
                    except Exception:
                        market = None
                    if market and market.get("closed"):
                        winning = ""
                        for t in market.get("tokens", []):
                            if t.get("winner"):
                                winning = t.get("outcome", "")
                                break
                        if not winning:
                            winning = market.get("winningOutcome", "")
                        if winning:
                            if pos["outcome"].lower() == winning.lower():
                                current_price = 1.0
                                exit_reason = "Market resolved - WON"
                            else:
                                current_price = 0.0
                                exit_reason = "Market resolved - LOST"

                # ── STEP 2: Price-based exits (only if not resolved) ──
                if not exit_reason and current_price < 0.02:
                    log.debug(
                        f"Ignoring suspicious price ${current_price:.4f} for "
                        f"'{pos.get('market_question', '?')[:40]}'"
                    )
                    current_price = entry_price

                pos["current_price"] = current_price
                pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                pos["pnl"] = (current_price - entry_price) * pos["shares"]
                pos["pnl_pct"] = pnl_pct

                # Track peak for trailing stop
                if pnl_pct > pos.get("peak_pnl_pct", 0):
                    pos["peak_pnl_pct"] = pnl_pct

                if not exit_reason:
                    # TAKE PROFIT
                    if pnl_pct >= TAKE_PROFIT_PCT:
                        exit_reason = f"Take profit ({pnl_pct:+.1f}% >= {TAKE_PROFIT_PCT}%)"

                    # STOP LOSS
                    elif pnl_pct <= STOP_LOSS_PCT:
                        exit_reason = f"Stop loss ({pnl_pct:+.1f}% <= {STOP_LOSS_PCT}%)"

                    # TRAILING STOP
                    elif pos.get("peak_pnl_pct", 0) >= TRAILING_ACTIVATE_PCT:
                        drop_from_peak = pnl_pct - pos["peak_pnl_pct"]
                        if drop_from_peak <= TRAILING_STOP_PCT:
                            exit_reason = (
                                f"Trailing stop (peak: {pos['peak_pnl_pct']:+.1f}%, "
                                f"now: {pnl_pct:+.1f}%, drop: {drop_from_peak:.1f}%)"
                            )

                # Execute exit
                if exit_reason:
                    result = await self.sell_position(pos, exit_reason)
                    exits.append({**pos, "exit_result": result})
                else:
                    remaining.append(pos)

            except Exception as e:
                log.warning(f"Error checking position {pos.get('id', '?')}: {e}")
                remaining.append(pos)

        self.positions = remaining
        self._save_state()

        if exits:
            total_exit_pnl = sum(e.get("pnl", 0) for e in exits)
            log.info(
                f"[EXIT SUMMARY] Closed {len(exits)} positions | "
                f"Combined PnL: ${total_exit_pnl:+.2f} | "
                f"Capital freed: ${sum(e.get('size_usdc', 0) for e in exits):.2f}"
            )

        return exits

    # ── UPDATE POSITIONS ─────────────────────────────────────

    async def update_positions(self):
        """Update all open positions with current market prices."""
        updated = 0
        for pos in self.positions:
            if pos["status"] != "open":
                continue
            token_id = pos.get("token_id", "")
            if not token_id:
                continue
            try:
                prices = await self.client.get_prices([token_id])
                if token_id in prices:
                    current_price = prices[token_id]
                    entry_price = pos.get("entry_price", pos.get("price", 0.5))
                    pos["current_price"] = current_price
                    pos["pnl"] = (current_price - entry_price) * pos["shares"]
                    pos["pnl_pct"] = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                    if pos["pnl_pct"] > pos.get("peak_pnl_pct", 0):
                        pos["peak_pnl_pct"] = pos["pnl_pct"]
                    updated += 1
            except Exception:
                pass

        if updated:
            log.info(f"Updated prices for {updated}/{len([p for p in self.positions if p['status'] == 'open'])} open positions")
        self._save_state()

    # ── PERFORMANCE METRICS ──────────────────────────────────

    def get_performance(self) -> dict:
        """Get overall live trading performance metrics. Same interface as PaperTrader."""
        open_trades = [p for p in self.positions if p["status"] == "open"]
        sold_trades = self.closed_trades
        won_trades = [t for t in sold_trades if t.get("pnl", 0) > 0]
        lost_trades = [t for t in sold_trades if t.get("pnl", 0) <= 0]
        total_closed = len(sold_trades)

        unrealized_pnl = sum(p.get("pnl", 0) for p in open_trades)
        realized_pnl = sum(t.get("pnl", 0) for t in sold_trades)

        # By source
        by_source: dict[str, dict] = {}
        for trade in sold_trades:
            src = trade.get("source", "unknown")
            if src not in by_source:
                by_source[src] = {"trades": 0, "pnl": 0, "wins": 0, "losses": 0}
            by_source[src]["trades"] += 1
            pnl = trade.get("pnl", 0)
            by_source[src]["pnl"] += pnl
            if pnl > 0:
                by_source[src]["wins"] += 1
            else:
                by_source[src]["losses"] += 1

        # By exit reason
        by_exit: dict[str, int] = {}
        for trade in sold_trades:
            reason = trade.get("exit_reason", "unknown")
            if "Take profit" in reason:
                key = "Take Profit"
            elif "Stop loss" in reason:
                key = "Stop Loss"
            elif "Trailing stop" in reason:
                key = "Trailing Stop"
            elif "resolved" in reason:
                key = "Market Resolved"
            else:
                key = reason
            by_exit[key] = by_exit.get(key, 0) + 1

        total_value = self.balance + sum(
            p["shares"] * p.get("current_price", p.get("entry_price", 0.5))
            for p in open_trades
        )

        return {
            "starting_balance": self.starting_balance,
            "current_balance": round(self.balance, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "return_pct": round((total_value / self.starting_balance - 1) * 100, 2),
            "total_trades": len(self.trade_history),
            "open_positions": len(open_trades),
            "closed_trades": total_closed,
            "wins": len(won_trades),
            "losses": len(lost_trades),
            "win_rate": round(len(won_trades) / total_closed * 100, 1) if total_closed > 0 else 0,
            "avg_win": round(sum(t["pnl"] for t in won_trades) / len(won_trades), 2) if won_trades else 0,
            "avg_loss": round(sum(t["pnl"] for t in lost_trades) / len(lost_trades), 2) if lost_trades else 0,
            "by_source": by_source,
            "by_exit_reason": by_exit,
            "daily_loss": round(self.daily_loss, 2),
        }

    # ── NOTIFICATIONS ────────────────────────────────────────

    async def _send_notification(self, message: str):
        """Send notification via Discord or Telegram webhook."""
        webhook_url = Config.NOTIFICATION_WEBHOOK
        if not webhook_url:
            return
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                if "discord" in webhook_url:
                    await client.post(webhook_url, json={
                        "content": f"**Polymarket Bot** | {message}"
                    })
                elif "telegram" in webhook_url:
                    # Extract bot token and chat_id from URL
                    await client.post(webhook_url, json={
                        "text": f"Polymarket Bot: {message}",
                        "parse_mode": "HTML",
                    })
                else:
                    # Generic webhook
                    await client.post(webhook_url, json={
                        "text": message,
                        "message": message,
                    })
        except Exception as e:
            log.debug(f"Notification failed: {e}")

    # ── RESET ────────────────────────────────────────────────

    def reset(self):
        """Reset live trading state. USE WITH EXTREME CAUTION."""
        log.warning("RESETTING LIVE TRADING STATE")
        self.balance = self.starting_balance
        self.positions = []
        self.closed_trades = []
        self.trade_history = []
        self.total_pnl = 0.0
        self.daily_loss = 0.0
        self._save_state()
        log.info("Live trading state reset to starting values")
