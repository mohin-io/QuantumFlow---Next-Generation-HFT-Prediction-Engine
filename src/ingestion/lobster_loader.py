"""
LOBSTER (Limit Order Book System - The Efficient Reconstructor) data loader.

LOBSTER provides historical limit order book data for NASDAQ stocks.
Data format:
- Message file: Contains events (new orders, cancellations, executions)
- Orderbook file: Contains orderbook snapshots after each event

Reference: https://lobsterdata.com/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, time as dt_time
import logging

from loguru import logger


@dataclass
class LOBSTERConfig:
    """Configuration for LOBSTER data loading."""

    levels: int = 10  # Number of price levels
    exchange: str = "NASDAQ"
    timezone: str = "America/New_York"


class LOBSTERLoader:
    """
    Loader for LOBSTER limit order book data.

    LOBSTER files format:
    - {TICKER}_{DATE}_{START}_{END}_message_{LEVELS}.csv: Events/messages
    - {TICKER}_{DATE}_{START}_{END}_orderbook_{LEVELS}.csv: Order book snapshots

    Message file columns:
    [Time, Event Type, Order ID, Size, Price, Direction]

    Event Types:
    1 = Submission of new limit order
    2 = Cancellation (partial or full)
    3 = Deletion (full cancellation)
    4 = Execution (visible)
    5 = Execution (hidden)
    6 = Cross trade
    7 = Trading halt

    Orderbook file columns (for N levels):
    [Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, ..., Ask Price N, Ask Size N, Bid Price N, Bid Size N]
    """

    def __init__(self, config: Optional[LOBSTERConfig] = None):
        self.config = config or LOBSTERConfig()

    def load_files(
        self,
        message_file: Path,
        orderbook_file: Path,
        normalize_time: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load LOBSTER message and orderbook files.

        Args:
            message_file: Path to message CSV file
            orderbook_file: Path to orderbook CSV file
            normalize_time: Convert timestamps to datetime

        Returns:
            Tuple of (messages_df, orderbook_df)
        """
        logger.info(f"Loading LOBSTER data from {message_file.parent}")

        # Load message file
        message_columns = ['timestamp', 'event_type', 'order_id', 'size', 'price', 'direction']
        messages_df = pd.read_csv(message_file, names=message_columns, header=None)

        # Load orderbook file
        orderbook_columns = self._generate_orderbook_columns(self.config.levels)
        orderbook_df = pd.read_csv(orderbook_file, names=orderbook_columns, header=None)

        if normalize_time:
            messages_df['timestamp'] = self._normalize_timestamps(messages_df['timestamp'])
            orderbook_df['timestamp'] = messages_df['timestamp'].values

        # Add metadata
        messages_df['exchange'] = self.config.exchange
        orderbook_df['exchange'] = self.config.exchange

        logger.info(f"Loaded {len(messages_df):,} messages and {len(orderbook_df):,} orderbook snapshots")

        return messages_df, orderbook_df

    def _generate_orderbook_columns(self, levels: int) -> List[str]:
        """Generate column names for orderbook DataFrame."""
        columns = []
        for i in range(1, levels + 1):
            columns.extend([
                f'ask_price_{i}',
                f'ask_size_{i}',
                f'bid_price_{i}',
                f'bid_size_{i}'
            ])
        return columns

    def _normalize_timestamps(self, timestamps: pd.Series) -> pd.Series:
        """
        Convert LOBSTER timestamps to datetime.

        LOBSTER timestamps are seconds from midnight.
        """
        # Assuming data is from a specific date (would be parsed from filename)
        base_date = pd.Timestamp.today().normalize()

        return pd.to_datetime(base_date) + pd.to_timedelta(timestamps, unit='s')

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse LOBSTER filename to extract metadata.

        Format: {TICKER}_{DATE}_{START}_{END}_message_{LEVELS}.csv
        Example: AAPL_2012-06-21_34200000_57600000_message_10.csv
        """
        parts = filename.replace('.csv', '').split('_')

        if len(parts) < 6:
            raise ValueError(f"Invalid LOBSTER filename format: {filename}")

        return {
            'ticker': parts[0],
            'date': parts[1],
            'start_time': parts[2],
            'end_time': parts[3],
            'file_type': parts[4],  # 'message' or 'orderbook'
            'levels': int(parts[5])
        }

    def reconstruct_order_book(
        self,
        messages_df: pd.DataFrame,
        orderbook_df: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Reconstruct standardized order book format from LOBSTER data.

        Returns DataFrame with columns:
        [timestamp, exchange, symbol, bids, asks, mid_price, spread]
        """
        end_idx = end_idx or len(orderbook_df)

        reconstructed = []

        for idx in range(start_idx, end_idx):
            row = orderbook_df.iloc[idx]

            # Extract bids and asks
            bids = []
            asks = []

            for i in range(1, self.config.levels + 1):
                bid_price = row[f'bid_price_{i}']
                bid_size = row[f'bid_size_{i}']
                ask_price = row[f'ask_price_{i}']
                ask_size = row[f'ask_size_{i}']

                if bid_price > 0 and bid_size > 0:
                    bids.append([bid_price, bid_size])
                if ask_price > 0 and ask_size > 0:
                    asks.append([ask_price, ask_size])

            if not bids or not asks:
                continue

            mid_price = (bids[0][0] + asks[0][0]) / 2
            spread = asks[0][0] - bids[0][0]

            reconstructed.append({
                'timestamp': row['timestamp'],
                'exchange': row['exchange'],
                'bids': bids,
                'asks': asks,
                'mid_price': mid_price,
                'spread': spread,
                'spread_bps': (spread / mid_price) * 10000 if mid_price > 0 else 0
            })

        return pd.DataFrame(reconstructed)

    def compute_event_statistics(self, messages_df: pd.DataFrame) -> Dict[str, any]:
        """Compute statistics on order book events."""
        event_counts = messages_df['event_type'].value_counts().to_dict()

        event_names = {
            1: 'new_limit_order',
            2: 'cancellation',
            3: 'deletion',
            4: 'execution_visible',
            5: 'execution_hidden',
            6: 'cross_trade',
            7: 'trading_halt'
        }

        stats = {
            'total_events': len(messages_df),
            'event_breakdown': {event_names.get(k, f'unknown_{k}'): v for k, v in event_counts.items()},
            'time_span_seconds': messages_df['timestamp'].max() - messages_df['timestamp'].min() if isinstance(messages_df['timestamp'].iloc[0], (int, float)) else None,
            'avg_events_per_second': len(messages_df) / (messages_df['timestamp'].max() - messages_df['timestamp'].min()) if isinstance(messages_df['timestamp'].iloc[0], (int, float)) else None
        }

        return stats

    def filter_by_time(
        self,
        messages_df: pd.DataFrame,
        orderbook_df: pd.DataFrame,
        start_time: Optional[dt_time] = None,
        end_time: Optional[dt_time] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data by time of day."""
        if start_time is None and end_time is None:
            return messages_df, orderbook_df

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(messages_df['timestamp']):
            raise ValueError("Timestamps must be datetime type. Use normalize_time=True when loading.")

        mask = pd.Series([True] * len(messages_df))

        if start_time:
            mask &= messages_df['timestamp'].dt.time >= start_time
        if end_time:
            mask &= messages_df['timestamp'].dt.time <= end_time

        filtered_messages = messages_df[mask].reset_index(drop=True)
        filtered_orderbook = orderbook_df[mask].reset_index(drop=True)

        logger.info(f"Filtered to {len(filtered_messages):,} events between {start_time} and {end_time}")

        return filtered_messages, filtered_orderbook


# Example usage and CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LOBSTER Data Loader and Analyzer")
    parser.add_argument("--message-file", type=Path, required=True, help="Path to LOBSTER message file")
    parser.add_argument("--orderbook-file", type=Path, required=True, help="Path to LOBSTER orderbook file")
    parser.add_argument("--levels", type=int, default=10, help="Number of price levels")
    parser.add_argument("--start-time", type=str, help="Start time (HH:MM:SS)")
    parser.add_argument("--end-time", type=str, help="End time (HH:MM:SS)")
    parser.add_argument("--stats", action="store_true", help="Print event statistics")

    args = parser.parse_args()

    # Initialize loader
    config = LOBSTERConfig(levels=args.levels)
    loader = LOBSTERLoader(config)

    # Load data
    messages_df, orderbook_df = loader.load_files(
        args.message_file,
        args.orderbook_file,
        normalize_time=True
    )

    # Filter by time if specified
    if args.start_time or args.end_time:
        start_t = datetime.strptime(args.start_time, "%H:%M:%S").time() if args.start_time else None
        end_t = datetime.strptime(args.end_time, "%H:%M:%S").time() if args.end_time else None
        messages_df, orderbook_df = loader.filter_by_time(messages_df, orderbook_df, start_t, end_t)

    # Print statistics
    if args.stats:
        stats = loader.compute_event_statistics(messages_df)
        print("\n" + "="*60)
        print("LOBSTER Data Statistics")
        print("="*60)
        print(f"Total Events: {stats['total_events']:,}")
        print("\nEvent Breakdown:")
        for event_type, count in stats['event_breakdown'].items():
            pct = (count / stats['total_events']) * 100
            print(f"  {event_type:20s}: {count:>10,} ({pct:>5.2f}%)")

        if stats['time_span_seconds']:
            print(f"\nTime Span: {stats['time_span_seconds']:.2f} seconds")
            print(f"Avg Events/Second: {stats['avg_events_per_second']:.2f}")

    # Reconstruct and display sample
    print("\n" + "="*60)
    print("Sample Order Book Snapshots (first 5)")
    print("="*60)
    reconstructed = loader.reconstruct_order_book(messages_df, orderbook_df, 0, 5)
    print(reconstructed[['timestamp', 'mid_price', 'spread', 'spread_bps']])
