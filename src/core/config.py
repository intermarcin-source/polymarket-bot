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
    POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")

    # Bot settings
    SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"
    MAX_BET_SIZE = float(os.getenv("MAX_BET_SIZE_USDC", "50"))
    MAX_PORTFOLIO_EXPOSURE = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_USDC", "500"))
    MIN_AI_CONFIDENCE = float(os.getenv("MIN_AI_CONFIDENCE", "0.75"))
    MIN_WHALE_FOLLOW_SIZE = float(os.getenv("MIN_WHALE_FOLLOW_SIZE_USDC", "1000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Strategy toggles (enable/disable individual strategies)
    ENABLE_WHALE_TRACKER = os.getenv("ENABLE_WHALE_TRACKER", "true").lower() == "true"
    ENABLE_ARBITRAGE = os.getenv("ENABLE_ARBITRAGE", "true").lower() == "true"
    ENABLE_MARKET_MAKER = os.getenv("ENABLE_MARKET_MAKER", "true").lower() == "true"
    ENABLE_MEAN_REVERSION = os.getenv("ENABLE_MEAN_REVERSION", "true").lower() == "true"
    ENABLE_NEGRISK = os.getenv("ENABLE_NEGRISK", "true").lower() == "true"

    # Arbitrage scanner settings
    ARB_MIN_EDGE = float(os.getenv("ARB_MIN_EDGE", "0.02"))  # 2% minimum profit
    ARB_MIN_LIQUIDITY = float(os.getenv("ARB_MIN_LIQUIDITY", "1000"))

    # Market maker settings
    MM_MIN_SPREAD = float(os.getenv("MM_MIN_SPREAD", "0.03"))  # 3 cent minimum spread
    MM_MIN_LIQUIDITY = float(os.getenv("MM_MIN_LIQUIDITY", "5000"))

    # Mean reversion settings
    MR_MIN_DROP = float(os.getenv("MR_MIN_DROP", "0.20"))  # 20% drop from avg to trigger
    MR_MIN_LIQUIDITY = float(os.getenv("MR_MIN_LIQUIDITY", "5000"))

    # NegRisk settings
    NEGRISK_MIN_EDGE = float(os.getenv("NEGRISK_MIN_EDGE", "0.03"))  # 3% deviation from 100%
    NEGRISK_MIN_LIQUIDITY = float(os.getenv("NEGRISK_MIN_LIQUIDITY", "2000"))

    # Live trading settings
    FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))  # 1 = Magic/email wallet
    MAX_DAILY_LOSS_USDC = float(os.getenv("MAX_DAILY_LOSS_USDC", "200"))
    KILL_SWITCH_FILE = os.getenv("KILL_SWITCH_FILE", "data/KILL_SWITCH")
    NOTIFICATION_WEBHOOK = os.getenv("NOTIFICATION_WEBHOOK", "")
    ORDER_TYPE = os.getenv("ORDER_TYPE", "FOK")  # FOK = market order, GTC = limit order
    MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "2.0"))

    # Polymarket API endpoints
    CLOB_API_URL = "https://clob.polymarket.com"
    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    STRAPI_API_URL = "https://strapi-matic.poly.market"

    # Known whale wallets (starting list - bot will discover more)
    WHALE_WALLETS = [
        "0x1a2b3c4d5e6f7890abcdef1234567890abcdef12",  # placeholder - we'll populate with real ones
    ]

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
