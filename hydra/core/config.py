"""HYDRA Configuration System."""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ExchangeConfig(BaseSettings):
    """Exchange API configuration."""
    model_config = SettingsConfigDict(env_prefix="")
    
    # Binance
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True
    
    # Bybit
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = True
    
    # Primary exchange
    primary_exchange: Literal["binance", "bybit"] = "binance"


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    model_config = SettingsConfigDict(env_prefix="")
    
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    
    llm_provider: Literal["openai", "anthropic", "groq"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"  # Groq's best model for trading
    
    # Rate limiting
    max_requests_per_minute: int = 30
    max_tokens_per_request: int = 4096


class DataConfig(BaseSettings):
    """Data provider configuration."""
    model_config = SettingsConfigDict(env_prefix="")
    
    # On-chain data
    glassnode_api_key: str = ""
    coinglass_api_key: str = ""
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./hydra.db"
    redis_url: str = "redis://localhost:6379/0"
    
    # Data retention
    ohlcv_retention_days: int = 365
    tick_retention_days: int = 30
    orderbook_retention_hours: int = 24


# Fixed trading universe - ONLY these pairs are permitted
PERMITTED_PAIRS: tuple[str, ...] = (
    "cmt_btcusdt",
    "cmt_ethusdt",
    "cmt_solusdt",
    "cmt_dogeusdt",
    "cmt_xrpusdt",
    "cmt_adausdt",
    "cmt_bnbusdt",
    "cmt_ltcusdt",
)

# Mapping for display names
PAIR_DISPLAY_NAMES: dict[str, str] = {
    "cmt_btcusdt": "BTC/USDT",
    "cmt_ethusdt": "ETH/USDT",
    "cmt_solusdt": "SOL/USDT",
    "cmt_dogeusdt": "DOGE/USDT",
    "cmt_xrpusdt": "XRP/USDT",
    "cmt_adausdt": "ADA/USDT",
    "cmt_bnbusdt": "BNB/USDT",
    "cmt_ltcusdt": "LTC/USDT",
}

# Pre-computed correlation groups for the fixed universe
CORRELATION_GROUPS: dict[str, list[str]] = {
    "btc_correlated": ["cmt_btcusdt", "cmt_ltcusdt"],  # High BTC correlation
    "eth_ecosystem": ["cmt_ethusdt", "cmt_adausdt"],   # Smart contract platforms
    "alt_majors": ["cmt_solusdt", "cmt_xrpusdt", "cmt_bnbusdt"],  # Major alts
    "meme_volatile": ["cmt_dogeusdt"],  # High volatility meme
}


class TradingConfig(BaseSettings):
    """Trading parameters configuration."""
    model_config = SettingsConfigDict(env_prefix="")
    
    trading_mode: Literal["live", "paper", "backtest"] = "paper"
    
    # Universe - FIXED to permitted pairs only
    symbols: list[str] = Field(default_factory=lambda: list(PERMITTED_PAIRS))
    
    # Position limits
    max_leverage: float = 10.0
    max_position_size_usd: float = 10000.0
    max_total_exposure_usd: float = 50000.0
    max_positions: int = 5  # Max 5 of 8 pairs at once
    
    @field_validator('symbols')
    @classmethod
    def validate_permitted_pairs(cls, v: list[str]) -> list[str]:
        """Ensure only permitted pairs are configured."""
        invalid = [s for s in v if s.lower() not in PERMITTED_PAIRS]
        if invalid:
            raise ValueError(
                f"Invalid pairs: {invalid}. "
                f"Only permitted: {list(PERMITTED_PAIRS)}"
            )
        # Normalize to lowercase
        return [s.lower() for s in v]
    
    # Risk per trade
    risk_per_trade_pct: float = 1.0
    max_drawdown_pct: float = 15.0
    
    # Execution
    default_order_type: Literal["limit", "market"] = "limit"
    max_slippage_bps: float = 10.0
    use_reduce_only: bool = True
    use_post_only: bool = True


class RiskConfig(BaseSettings):
    """Risk management configuration."""
    model_config = SettingsConfigDict(env_prefix="RISK_")
    
    # Position sizing
    kelly_fraction: float = 0.25  # Quarter Kelly
    risk_per_trade_pct: float = 1.0  # Risk 1% of equity per trade
    min_confidence_threshold: float = 0.6
    
    # Leverage governance (max 20x allowed)
    base_leverage: float = 3.0
    max_leverage: float = 20.0  # Maximum 20x leverage
    leverage_decay_per_sigma: float = 0.5
    
    # Position management
    max_holding_time_hours: float = 24.0  # Default max hold time
    thesis_review_interval_minutes: int = 5  # How often to check thesis
    force_exit_loss_pct: float = 5.0  # Force exit at -5% unrealized
    take_profit_pct: float = 3.0  # Default take profit at +3%
    
    # Funding awareness
    max_funding_rate_to_enter: float = 0.001  # 0.1%
    funding_impact_weight: float = 0.3
    
    # Liquidation buffers
    min_liquidation_distance_pct: float = 20.0
    
    # Kill switches
    max_hourly_drawdown_pct: float = 3.0
    max_daily_drawdown_pct: float = 8.0
    max_correlation_exposure: float = 0.8
    
    # Model disagreement
    max_model_disagreement: float = 0.4
    min_agents_agreement: int = 3  # Out of 4


class ModelConfig(BaseSettings):
    """ML/AI model configuration."""
    model_config = SettingsConfigDict(env_prefix="MODEL_")
    
    # Transformer
    transformer_hidden_size: int = 256
    transformer_num_layers: int = 4
    transformer_num_heads: int = 8
    transformer_sequence_length: int = 100
    
    # RL
    rl_learning_rate: float = 3e-4
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95
    rl_clip_range: float = 0.2
    
    # Training
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Model paths
    models_dir: str = "./models"
    checkpoint_interval: int = 1000


class SystemConfig(BaseSettings):
    """System configuration."""
    model_config = SettingsConfigDict(env_prefix="")
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    
    # Timeframes to track
    timeframes: list[str] = Field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h"
    ])
    
    # Update intervals (seconds)
    orderbook_update_interval: float = 1.0
    ohlcv_update_interval: float = 5.0
    funding_update_interval: float = 60.0
    oi_update_interval: float = 30.0
    
    # Decision cycle
    decision_interval_seconds: int = 60


class HydraConfig(BaseSettings):
    """Master HYDRA configuration."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "HydraConfig":
        """Load configuration from environment."""
        if env_file:
            return cls(_env_file=env_file)
        return cls()

    def validate_for_live_trading(self) -> list[str]:
        """Validate configuration for live trading. Returns list of errors."""
        errors = []
        
        if self.trading.trading_mode == "live":
            if not self.exchange.binance_api_key and not self.exchange.bybit_api_key:
                errors.append("No exchange API keys configured for live trading")
            
            if self.trading.max_leverage > 20:
                errors.append(f"Max leverage {self.trading.max_leverage} is dangerously high")
            
            if self.risk.min_liquidation_distance_pct < 10:
                errors.append("Liquidation distance buffer is too small")
        
        if not self.llm.openai_api_key and not self.llm.anthropic_api_key and not self.llm.groq_api_key:
            errors.append("No LLM API key configured - LLM agent will be disabled")
        
        return errors
