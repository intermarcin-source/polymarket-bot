import os
from dotenv import load_dotenv
from pathlib import Path

_env_path = (Path(__file__).parent.parent.parent / ".env").resolve()
load_dotenv(_env_path, override=True)


class Config:
    # Wallet
    WALLET_ADDRESS = os.getenv("POLYMARKET_WALLET_ADDRESS", "")
    PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    API_KEY = os.getenv("POLYMARKET_API_KEY", "")
    API_SECRET = os.getenv("POLYMARKET_API_SECRET", "")
    API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "")

    # Blockchain
    POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")

    # Bot settings
    SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # BTC Sniper settings
    BASE_BET_SIZE = float(os.getenv("BASE_BET_SIZE_USDC", "50"))
    PROFIT_SCALE_PCT = float(os.getenv("PROFIT_SCALE_PCT", "10"))  # Bet extra 10% of cumulative profit
    MAX_BET_PCT_OF_BALANCE = float(os.getenv("MAX_BET_PCT_OF_BALANCE", "5"))  # Never bet >5% of balance
    MIN_PRICE_DELTA_PCT = float(os.getenv("MIN_PRICE_DELTA_PCT", "0.03"))  # Min BTC move to trigger
    ENTRY_SECONDS_BEFORE_CLOSE = int(os.getenv("ENTRY_SECONDS_BEFORE_CLOSE", "10"))

    # Live trading settings
    FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
    MAX_DAILY_LOSS_USDC = float(os.getenv("MAX_DAILY_LOSS_USDC", "200"))
    KILL_SWITCH_FILE = os.getenv("KILL_SWITCH_FILE", "data/KILL_SWITCH")
    NOTIFICATION_WEBHOOK = os.getenv("NOTIFICATION_WEBHOOK", "")
    MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "5.0"))  # Wider for fast markets

    # Polymarket API endpoints
    CLOB_API_URL = "https://clob.polymarket.com"
    GAMMA_API_URL = "https://gamma-api.polymarket.com"

    @classmethod
    def validate(cls):
        errors = []
        if not cls.WALLET_ADDRESS:
            errors.append("POLYMARKET_WALLET_ADDRESS is not set")
        if not cls.SIMULATION_MODE:
            if not cls.PRIVATE_KEY or cls.PRIVATE_KEY == "your_private_key_here":
                errors.append("Valid POLYMARKET_PRIVATE_KEY required for live trading")
            if not cls.FUNDER_ADDRESS:
                errors.append("POLYMARKET_FUNDER_ADDRESS required for live trading")
        return errors
