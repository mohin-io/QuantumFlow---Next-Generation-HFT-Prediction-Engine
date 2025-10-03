"""
Database models and schemas for storing order book data.

Uses SQLAlchemy ORM with TimescaleDB for time-series optimization.
"""

from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    BigInteger,
    DateTime,
    JSON,
    Index,
    Boolean,
    Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class OrderBookSnapshot(Base):
    """
    Table for storing order book snapshots.

    Optimized as TimescaleDB hypertable for time-series queries.
    """

    __tablename__ = "order_book_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Order book data stored as JSONB for flexibility
    bids = Column(JSONB, nullable=False)  # [[price, size], ...]
    asks = Column(JSONB, nullable=False)  # [[price, size], ...]

    # Pre-computed metrics for faster queries
    mid_price = Column(Float)
    spread = Column(Float)
    spread_bps = Column(Float)
    bid_volume = Column(Float)
    ask_volume = Column(Float)

    # Sequence number for ordering (exchange-specific)
    sequence = Column(BigInteger)

    # Indexes
    __table_args__ = (
        Index('ix_snapshot_time_exchange_symbol', 'timestamp', 'exchange', 'symbol'),
        Index('ix_snapshot_exchange_symbol', 'exchange', 'symbol'),
    )

    def __repr__(self):
        return f"<OrderBookSnapshot(timestamp={self.timestamp}, exchange={self.exchange}, symbol={self.symbol})>"


class Trade(Base):
    """Table for storing executed trades."""

    __tablename__ = "trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    trade_id = Column(String(100))  # Exchange trade ID
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    side = Column(String(10))  # buy or sell
    is_maker = Column(Boolean)

    __table_args__ = (
        Index('ix_trade_time_exchange_symbol', 'timestamp', 'exchange', 'symbol'),
    )


class ComputedFeature(Base):
    """Table for storing computed microstructure features."""

    __tablename__ = "computed_features"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Order Flow Imbalance
    ofi_l1 = Column(Float)
    ofi_l5 = Column(Float)
    ofi_l10 = Column(Float)

    # Micro-price and volume metrics
    micro_price = Column(Float)
    volume_imbalance = Column(Float)
    depth_imbalance = Column(Float)

    # Queue dynamics
    cancellation_ratio = Column(Float)
    arrival_rate = Column(Float)

    # Volatility
    realized_volatility_20 = Column(Float)
    realized_volatility_50 = Column(Float)
    realized_volatility_100 = Column(Float)

    # All features as JSON for flexibility
    features_json = Column(JSONB)

    __table_args__ = (
        Index('ix_feature_time_exchange_symbol', 'timestamp', 'exchange', 'symbol'),
    )


class ModelPrediction(Base):
    """Table for storing model predictions."""

    __tablename__ = "model_predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))

    # Predictions
    prediction_class = Column(String(20))  # up, down, flat
    prediction_probabilities = Column(JSONB)  # {up: 0.4, down: 0.3, flat: 0.3}

    # Actual outcome (filled later for evaluation)
    actual_class = Column(String(20))
    actual_price_change = Column(Float)

    # Confidence and metadata
    confidence = Column(Float)
    prediction_horizon = Column(Integer)  # ticks ahead

    __table_args__ = (
        Index('ix_prediction_time_model', 'timestamp', 'model_name'),
    )


class BacktestResult(Base):
    """Table for storing backtesting results."""

    __tablename__ = "backtest_results"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(100), nullable=False, unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    model_name = Column(String(100), nullable=False)
    model_config = Column(JSONB)

    # Data parameters
    exchange = Column(String(50))
    symbol = Column(String(20))
    start_date = Column(DateTime)
    end_date = Column(DateTime)

    # Performance metrics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)

    # Returns
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    calmar_ratio = Column(Float)

    # Classification metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    # Transaction costs
    total_cost = Column(Float)
    cost_per_trade = Column(Float)

    # Full results as JSON
    detailed_results = Column(JSONB)

    def __repr__(self):
        return f"<BacktestResult(run_id={self.run_id}, model={self.model_name}, sharpe={self.sharpe_ratio})>"


# Database connection and utilities

class DatabaseManager:
    """Manager for database connections and operations."""

    def __init__(self, connection_string: str = None):
        if connection_string is None:
            # Build from environment variables
            connection_string = self._build_connection_string()

        self.engine = create_engine(connection_string, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables."""
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "hft_orderbook")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        print("Database tables created successfully")

    def create_hypertables(self, session: Session):
        """
        Convert tables to TimescaleDB hypertables for time-series optimization.

        Note: Requires TimescaleDB extension to be enabled in PostgreSQL.
        """
        hypertables = [
            ("order_book_snapshots", "timestamp"),
            ("trades", "timestamp"),
            ("computed_features", "timestamp"),
            ("model_predictions", "timestamp"),
        ]

        for table_name, time_column in hypertables:
            try:
                sql = f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE);"
                session.execute(sql)
                session.commit()
                print(f"Created hypertable: {table_name}")
            except Exception as e:
                print(f"Error creating hypertable {table_name}: {e}")
                session.rollback()

    def create_retention_policy(self, session: Session, table_name: str, retention_days: int = 90):
        """Create data retention policy for TimescaleDB hypertable."""
        try:
            sql = f"""
            SELECT add_retention_policy('{table_name}', INTERVAL '{retention_days} days');
            """
            session.execute(sql)
            session.commit()
            print(f"Created retention policy for {table_name}: {retention_days} days")
        except Exception as e:
            print(f"Error creating retention policy: {e}")
            session.rollback()

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def close(self):
        """Close database connections."""
        self.engine.dispose()


# Helper functions for data insertion

def insert_order_book_snapshot(
    session: Session,
    timestamp: datetime,
    exchange: str,
    symbol: str,
    bids: List[List[float]],
    asks: List[List[float]],
    sequence: int = None
) -> OrderBookSnapshot:
    """Insert an order book snapshot into the database."""

    # Calculate metrics
    mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None
    spread = asks[0][0] - bids[0][0] if bids and asks else None
    spread_bps = (spread / mid_price * 10000) if mid_price and spread else None
    bid_volume = sum(b[1] for b in bids)
    ask_volume = sum(a[1] for a in asks)

    snapshot = OrderBookSnapshot(
        timestamp=timestamp,
        exchange=exchange,
        symbol=symbol,
        bids=bids,
        asks=asks,
        mid_price=mid_price,
        spread=spread,
        spread_bps=spread_bps,
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        sequence=sequence
    )

    session.add(snapshot)
    return snapshot


# CLI for database setup
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database Setup and Management")
    parser.add_argument("--create-tables", action="store_true", help="Create database tables")
    parser.add_argument("--create-hypertables", action="store_true", help="Convert to TimescaleDB hypertables")
    parser.add_argument("--retention-days", type=int, default=90, help="Data retention in days")

    args = parser.parse_args()

    # Initialize database manager
    db_manager = DatabaseManager()

    if args.create_tables:
        print("Creating database tables...")
        db_manager.create_tables()

    if args.create_hypertables:
        print("Creating TimescaleDB hypertables...")
        session = db_manager.get_session()
        db_manager.create_hypertables(session)

        if args.retention_days:
            print(f"Setting retention policy: {args.retention_days} days")
            for table in ["order_book_snapshots", "trades", "computed_features", "model_predictions"]:
                db_manager.create_retention_policy(session, table, args.retention_days)

        session.close()

    print("Database setup complete!")
