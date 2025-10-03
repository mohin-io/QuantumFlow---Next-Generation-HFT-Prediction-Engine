"""
Centralized Configuration Management with Pydantic

Provides type-safe, validated configuration management for all components
with environment-specific configs and environment variable overrides.

FEATURES:
- Type-safe configuration with Pydantic
- Environment-specific configs (dev/staging/prod)
- Environment variable overrides
- Validation on load
- Secrets management
- Configuration versioning
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator, SecretStr
from enum import Enum


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# Database Configurations
# ============================================================================


class DatabaseConfig(BaseModel):
    """PostgreSQL/TimescaleDB configuration."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="hft_orderbook", description="Database name")
    user: str = Field(default="hft_user", description="Database user")
    password: SecretStr = Field(default="", description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max pool overflow")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    echo: bool = Field(default=False, description="Echo SQL statements")

    class Config:
        env_prefix = "DB_"

    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class InfluxDBConfig(BaseModel):
    """InfluxDB configuration for tick data."""

    url: str = Field(default="http://localhost:8086", description="InfluxDB URL")
    token: SecretStr = Field(default="", description="InfluxDB token")
    org: str = Field(default="hft-org", description="Organization")
    bucket: str = Field(default="orderbook-data", description="Bucket name")
    timeout: int = Field(default=10000, ge=1000, description="Request timeout (ms)")

    class Config:
        env_prefix = "INFLUX_"


class RedisConfig(BaseModel):
    """Redis configuration for caching."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[SecretStr] = Field(None, description="Redis password")
    max_connections: int = Field(
        default=50, ge=1, description="Max connections in pool"
    )
    socket_timeout: int = Field(
        default=5, ge=1, description="Socket timeout in seconds"
    )
    decode_responses: bool = Field(default=True, description="Decode responses to str")

    class Config:
        env_prefix = "REDIS_"


# ============================================================================
# Streaming Configurations
# ============================================================================


class KafkaConfig(BaseModel):
    """Apache Kafka configuration."""

    bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], description="Kafka bootstrap servers"
    )
    topic_orderbook: str = Field(
        default="orderbook-updates", description="Order book updates topic"
    )
    topic_features: str = Field(
        default="computed-features", description="Computed features topic"
    )
    topic_predictions: str = Field(
        default="model-predictions", description="Model predictions topic"
    )
    consumer_group: str = Field(
        default="hft-consumers", description="Consumer group ID"
    )
    auto_offset_reset: str = Field(
        default="latest",
        regex="^(earliest|latest)$",
        description="Auto offset reset strategy",
    )

    class Config:
        env_prefix = "KAFKA_"


# ============================================================================
# Feature Engineering Configurations
# ============================================================================


class OFIConfig(BaseModel):
    """Order Flow Imbalance configuration."""

    levels: List[int] = Field(
        default=[1, 5, 10], description="Price levels for OFI calculation"
    )
    windows: List[int] = Field(
        default=[10, 50, 100], description="Rolling window sizes for OFI statistics"
    )
    enable_multi_level: bool = Field(
        default=True, description="Enable multi-level OFI computation"
    )


class MicroPriceConfig(BaseModel):
    """Micro-price calculation configuration."""

    depth_levels: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of depth levels for weighted calculations",
    )
    use_adaptive: bool = Field(
        default=True, description="Use adaptive fair value estimator"
    )
    alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Smoothing factor for adaptive estimator",
    )


class VolatilityConfig(BaseModel):
    """Realized volatility configuration."""

    estimators: List[str] = Field(
        default=["simple", "parkinson", "garman_klass"],
        description="Volatility estimators to use",
    )
    windows: List[int] = Field(
        default=[20, 50, 100], description="Rolling windows for volatility"
    )


class FeatureConfig(BaseModel):
    """Feature pipeline configuration."""

    ofi: OFIConfig = Field(default_factory=OFIConfig)
    micro_price: MicroPriceConfig = Field(default_factory=MicroPriceConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    enable_volume_profiles: bool = Field(default=True)
    enable_queue_dynamics: bool = Field(default=True)
    sequence_length: int = Field(default=50, ge=10, le=500)


# ============================================================================
# Model Configurations
# ============================================================================


class LSTMModelConfig(BaseModel):
    """LSTM model configuration."""

    input_size: int = Field(default=64, ge=1, description="Input feature size")
    hidden_size: int = Field(default=128, ge=1, description="Hidden layer size")
    num_layers: int = Field(default=2, ge=1, le=10, description="Number of LSTM layers")
    num_classes: int = Field(default=3, ge=2, description="Number of output classes")
    dropout: float = Field(default=0.3, ge=0.0, le=0.9, description="Dropout rate")
    bidirectional: bool = Field(default=False, description="Use bidirectional LSTM")
    attention: bool = Field(default=False, description="Use attention mechanism")


class TransformerModelConfig(BaseModel):
    """Transformer model configuration."""

    input_size: int = Field(default=64, ge=1, description="Input feature size")
    d_model: int = Field(default=128, ge=1, description="Model dimension")
    nhead: int = Field(default=8, ge=1, description="Number of attention heads")
    num_encoder_layers: int = Field(
        default=4, ge=1, le=12, description="Encoder layers"
    )
    dim_feedforward: int = Field(default=512, ge=1, description="Feedforward dimension")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")
    num_classes: int = Field(default=3, ge=2, description="Number of output classes")

    @validator("d_model")
    def d_model_divisible_by_nhead(cls, v, values):
        """Ensure d_model is divisible by nhead."""
        if "nhead" in values and v % values["nhead"] != 0:
            raise ValueError(
                f"d_model ({v}) must be divisible by nhead ({values['nhead']})"
            )
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model name")
    type: str = Field(
        ..., regex="^(lstm|transformer|bayesian|ensemble)$", description="Model type"
    )
    checkpoint_path: Optional[Path] = Field(
        None, description="Path to model checkpoint"
    )
    device: str = Field(
        default="cuda",
        regex="^(cpu|cuda|cuda:[0-9]+)$",
        description="Device for inference",
    )
    lstm: Optional[LSTMModelConfig] = None
    transformer: Optional[TransformerModelConfig] = None

    @validator("checkpoint_path")
    def checkpoint_exists(cls, v):
        """Validate checkpoint exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"Checkpoint file not found: {v}")
        return v


# ============================================================================
# API Configurations
# ============================================================================


class APIConfig(BaseModel):
    """FastAPI service configuration."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=4, ge=1, le=32, description="Number of workers")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    rate_limit: str = Field(
        default="10/minute", description="Rate limit (requests/period)"
    )
    request_timeout: int = Field(
        default=5, ge=1, le=60, description="Request timeout in seconds"
    )
    cache_ttl: int = Field(default=60, ge=0, description="Cache TTL in seconds")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")


# ============================================================================
# Backtesting Configurations
# ============================================================================


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    initial_capital: float = Field(
        default=100000.0, gt=0, description="Initial capital"
    )
    position_size: float = Field(
        default=0.1, gt=0, le=1, description="Position size as fraction of capital"
    )
    transaction_cost_bps: float = Field(
        default=5.0, ge=0, description="Transaction cost in basis points"
    )
    slippage_bps: float = Field(
        default=2.0, ge=0, description="Slippage in basis points"
    )
    max_holding_period: int = Field(
        default=100, ge=1, description="Maximum holding period in ticks"
    )
    confidence_threshold: float = Field(
        default=0.6, gt=0, lt=1, description="Minimum confidence for trade execution"
    )


# ============================================================================
# Main Application Configuration
# ============================================================================


class AppConfig(BaseModel):
    """Main application configuration."""

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Global log level")

    # Components
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    influxdb: InfluxDBConfig = Field(default_factory=InfluxDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: List[ModelConfig] = Field(default_factory=list)
    api: APIConfig = Field(default_factory=APIConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    models_dir: Path = Field(
        default=Path("models/saved"), description="Models directory"
    )
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "AppConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            AppConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Override with environment variables
        data = cls._apply_env_overrides(data)

        return cls(**data)

    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to config.

        Environment variables follow the pattern: COMPONENT_KEY
        Examples:
            DB_HOST=localhost
            REDIS_PORT=6379
            API_PORT=8000
        """
        # Database overrides
        if "database" not in data:
            data["database"] = {}

        for key in ["host", "port", "database", "user", "password"]:
            env_key = f"DB_{key.upper()}"
            if env_key in os.environ:
                data["database"][key] = os.environ[env_key]

        # Redis overrides
        if "redis" not in data:
            data["redis"] = {}

        for key in ["host", "port", "db", "password"]:
            env_key = f"REDIS_{key.upper()}"
            if env_key in os.environ:
                data["redis"][key] = os.environ[env_key]

        # API overrides
        if "api" not in data:
            data["api"] = {}

        for key in ["host", "port", "workers"]:
            env_key = f"API_{key.upper()}"
            if env_key in os.environ:
                data["api"][key] = os.environ[env_key]

        # Environment override
        if "ENVIRONMENT" in os.environ:
            data["environment"] = os.environ["ENVIRONMENT"]

        return data

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate environment
        if self.environment not in Environment:
            raise ValueError(f"Invalid environment: {self.environment}")

        # Validate paths exist or can be created
        for path_attr in ["data_dir", "models_dir", "logs_dir"]:
            path = getattr(self, path_attr)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create {path_attr} at {path}: {e}")

        # Validate production settings
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("Debug mode should be disabled in production")
            if self.api.reload:
                raise ValueError("Auto-reload should be disabled in production")
            if "*" in self.api.cors_origins:
                raise ValueError("CORS origins should be restricted in production")

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            output_path: Output YAML file path
        """
        output_path = Path(output_path)

        # Convert to dict, excluding secrets
        config_dict = self.dict(
            exclude={"database": {"password"}, "redis": {"password"}}
        )

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model."""
        for model in self.models:
            if model.name == model_name:
                return model
        return None


# ============================================================================
# Helper Functions
# ============================================================================


def load_config(
    config_path: Optional[Union[str, Path]] = None, environment: Optional[str] = None
) -> AppConfig:
    """
    Load application configuration.

    Args:
        config_path: Path to config file. If None, uses default based on environment.
        environment: Environment name (development/staging/production)

    Returns:
        AppConfig instance

    Example:
        >>> config = load_config("configs/production.yaml")
        >>> print(config.database.host)
        >>> print(config.api.port)
    """
    # Determine environment
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    # Determine config path
    if config_path is None:
        config_path = Path("configs") / f"{environment}.yaml"

    # Load and validate config
    config = AppConfig.from_yaml(config_path)
    config.validate()

    return config


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create example configurations
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Development config
    dev_config = AppConfig(
        environment=Environment.DEVELOPMENT,
        debug=True,
        log_level=LogLevel.DEBUG,
        api=APIConfig(reload=True, workers=1),
    )
    dev_config.to_yaml(configs_dir / "development.yaml")
    print("Created configs/development.yaml")

    # Production config
    prod_config = AppConfig(
        environment=Environment.PRODUCTION,
        debug=False,
        log_level=LogLevel.INFO,
        api=APIConfig(reload=False, workers=4, cors_origins=["https://yourdomain.com"]),
    )
    prod_config.to_yaml(configs_dir / "production.yaml")
    print("Created configs/production.yaml")

    # Test loading
    loaded_config = load_config(configs_dir / "development.yaml")
    print(f"\nLoaded config: {loaded_config.environment}")
    print(f"Database: {loaded_config.database.host}:{loaded_config.database.port}")
    print(f"API: {loaded_config.api.host}:{loaded_config.api.port}")
