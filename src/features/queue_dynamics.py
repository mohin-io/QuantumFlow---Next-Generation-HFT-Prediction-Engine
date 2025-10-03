"""
Queue Dynamics and Order Book Event Analysis

Tracks the dynamics of orders in the limit order book queue:
- Order arrival rates
- Cancellation ratios
- Queue position changes
- Time-to-fill estimates
- Order book intensity measures

These metrics capture the behavior of market participants and provide
insights into aggressive vs passive trading activity.

References:
- Cont, R., & De Larrard, A. (2013). Price dynamics in a Markovian limit order market
- Huang, W., Lehalle, C. A., & Rosenbaum, M. (2015). Simulating and analyzing order book data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Deque
from dataclasses import dataclass
from collections import deque, defaultdict


@dataclass
class OrderBookEvent:
    """Represents an order book event."""
    timestamp: float
    event_type: str  # 'new', 'cancel', 'modify', 'execute'
    side: str  # 'bid' or 'ask'
    price: float
    volume: float
    level: int = 0  # Price level (0 = best price)


@dataclass
class QueueMetrics:
    """Container for queue dynamics metrics."""

    # Arrival rates
    bid_arrival_rate: float
    ask_arrival_rate: float
    total_arrival_rate: float

    # Cancellation metrics
    bid_cancel_ratio: float
    ask_cancel_ratio: float
    total_cancel_ratio: float

    # Order book intensity
    order_book_intensity: float  # Total event rate

    # Queue position metrics
    avg_queue_depth_bid: float
    avg_queue_depth_ask: float

    # Time-based metrics
    avg_time_between_events: float


class QueueDynamicsCalculator:
    """
    Calculator for order book queue dynamics.

    Tracks order arrivals, cancellations, and executions to compute
    metrics about market participant behavior.
    """

    def __init__(
        self,
        window_size: int = 100,
        time_window_seconds: float = 60.0
    ):
        """
        Initialize queue dynamics calculator.

        Args:
            window_size: Number of events to track
            time_window_seconds: Time window for rate calculations
        """
        self.window_size = window_size
        self.time_window_seconds = time_window_seconds

        # Event history
        self.events: Deque[OrderBookEvent] = deque(maxlen=window_size)
        self.event_times: Deque[float] = deque(maxlen=window_size)

        # Counters
        self.bid_arrivals = 0
        self.ask_arrivals = 0
        self.bid_cancels = 0
        self.ask_cancels = 0
        self.bid_executions = 0
        self.ask_executions = 0

    def add_event(self, event: OrderBookEvent):
        """Add an order book event to history."""
        self.events.append(event)
        self.event_times.append(event.timestamp)

        # Update counters
        if event.side == 'bid':
            if event.event_type == 'new':
                self.bid_arrivals += 1
            elif event.event_type == 'cancel':
                self.bid_cancels += 1
            elif event.event_type == 'execute':
                self.bid_executions += 1
        else:  # ask
            if event.event_type == 'new':
                self.ask_arrivals += 1
            elif event.event_type == 'cancel':
                self.ask_cancels += 1
            elif event.event_type == 'execute':
                self.ask_executions += 1

    def compute_metrics(self) -> QueueMetrics:
        """Compute queue dynamics metrics."""
        if len(self.events) < 2:
            return self._initialize_metrics()

        # Time-based calculations
        time_span = self.event_times[-1] - self.event_times[0]
        if time_span <= 0:
            return self._initialize_metrics()

        # Arrival rates (events per second)
        bid_arrival_rate = self.bid_arrivals / time_span
        ask_arrival_rate = self.ask_arrivals / time_span
        total_arrival_rate = (self.bid_arrivals + self.ask_arrivals) / time_span

        # Cancellation ratios
        total_bid_events = self.bid_arrivals + self.bid_cancels + self.bid_executions
        total_ask_events = self.ask_arrivals + self.ask_cancels + self.ask_executions

        bid_cancel_ratio = self.bid_cancels / total_bid_events if total_bid_events > 0 else 0
        ask_cancel_ratio = self.ask_cancels / total_ask_events if total_ask_events > 0 else 0
        total_cancel_ratio = (self.bid_cancels + self.ask_cancels) / (total_bid_events + total_ask_events) if (total_bid_events + total_ask_events) > 0 else 0

        # Order book intensity (total event rate)
        order_book_intensity = len(self.events) / time_span

        # Average queue depth (simplified - using volume at different levels)
        bid_depths = [e.volume for e in self.events if e.side == 'bid']
        ask_depths = [e.volume for e in self.events if e.side == 'ask']

        avg_queue_depth_bid = np.mean(bid_depths) if bid_depths else 0
        avg_queue_depth_ask = np.mean(ask_depths) if ask_depths else 0

        # Time between events
        time_diffs = np.diff(list(self.event_times))
        avg_time_between_events = np.mean(time_diffs) if len(time_diffs) > 0 else 0

        return QueueMetrics(
            bid_arrival_rate=bid_arrival_rate,
            ask_arrival_rate=ask_arrival_rate,
            total_arrival_rate=total_arrival_rate,
            bid_cancel_ratio=bid_cancel_ratio,
            ask_cancel_ratio=ask_cancel_ratio,
            total_cancel_ratio=total_cancel_ratio,
            order_book_intensity=order_book_intensity,
            avg_queue_depth_bid=avg_queue_depth_bid,
            avg_queue_depth_ask=avg_queue_depth_ask,
            avg_time_between_events=avg_time_between_events
        )

    def _initialize_metrics(self) -> QueueMetrics:
        """Return zero-initialized metrics."""
        return QueueMetrics(
            bid_arrival_rate=0.0,
            ask_arrival_rate=0.0,
            total_arrival_rate=0.0,
            bid_cancel_ratio=0.0,
            ask_cancel_ratio=0.0,
            total_cancel_ratio=0.0,
            order_book_intensity=0.0,
            avg_queue_depth_bid=0.0,
            avg_queue_depth_ask=0.0,
            avg_time_between_events=0.0
        )

    def reset(self):
        """Reset the calculator state."""
        self.events.clear()
        self.event_times.clear()
        self.bid_arrivals = 0
        self.ask_arrivals = 0
        self.bid_cancels = 0
        self.ask_cancels = 0
        self.bid_executions = 0
        self.ask_executions = 0


class OrderBookIntensityAnalyzer:
    """
    Analyzes the intensity of order book events using Hawkes process approach.

    Captures the self-exciting nature of order arrivals and cancellations.
    """

    def __init__(self, decay_factor: float = 0.1):
        """
        Initialize intensity analyzer.

        Args:
            decay_factor: Exponential decay rate for intensity calculation
        """
        self.decay_factor = decay_factor
        self.base_intensity = 0.0
        self.current_intensity = 0.0
        self.last_event_time = None

    def update_intensity(self, timestamp: float, event_count: int = 1):
        """
        Update intensity based on new events.

        Simplified Hawkes process: λ(t) = μ + α * exp(-β * Δt)
        """
        if self.last_event_time is None:
            self.current_intensity = self.base_intensity + event_count
        else:
            time_diff = timestamp - self.last_event_time
            decay = np.exp(-self.decay_factor * time_diff)
            self.current_intensity = self.base_intensity + (self.current_intensity - self.base_intensity) * decay + event_count

        self.last_event_time = timestamp
        return self.current_intensity

    def get_intensity(self) -> float:
        """Get current intensity."""
        return self.current_intensity


def compute_queue_metrics_from_snapshots(
    snapshots: List[Dict],
    window_size: int = 100
) -> pd.DataFrame:
    """
    Compute queue dynamics from order book snapshots.

    This is a simplified version that infers events from snapshot changes.
    In production, you'd use actual message/event data.

    Args:
        snapshots: List of order book snapshots
        window_size: Window size for calculations

    Returns:
        DataFrame with queue metrics
    """
    calculator = QueueDynamicsCalculator(window_size=window_size)
    metrics_list = []

    prev_snapshot = None

    for snapshot in snapshots:
        timestamp = snapshot.get('timestamp', 0)
        bids = snapshot.get('bids', [])
        asks = snapshot.get('asks', [])

        if prev_snapshot is not None:
            # Infer events from snapshot changes
            prev_bids = prev_snapshot.get('bids', [])
            prev_asks = prev_snapshot.get('asks', [])

            # Check for new orders (simplified: compare volumes)
            if bids and prev_bids:
                bid_vol_change = bids[0][1] - prev_bids[0][1]
                if bid_vol_change > 0:
                    event = OrderBookEvent(
                        timestamp=timestamp,
                        event_type='new',
                        side='bid',
                        price=bids[0][0],
                        volume=bid_vol_change,
                        level=0
                    )
                    calculator.add_event(event)
                elif bid_vol_change < 0:
                    event = OrderBookEvent(
                        timestamp=timestamp,
                        event_type='cancel',
                        side='bid',
                        price=bids[0][0],
                        volume=abs(bid_vol_change),
                        level=0
                    )
                    calculator.add_event(event)

            if asks and prev_asks:
                ask_vol_change = asks[0][1] - prev_asks[0][1]
                if ask_vol_change > 0:
                    event = OrderBookEvent(
                        timestamp=timestamp,
                        event_type='new',
                        side='ask',
                        price=asks[0][0],
                        volume=ask_vol_change,
                        level=0
                    )
                    calculator.add_event(event)
                elif ask_vol_change < 0:
                    event = OrderBookEvent(
                        timestamp=timestamp,
                        event_type='cancel',
                        side='ask',
                        price=asks[0][0],
                        volume=abs(ask_vol_change),
                        level=0
                    )
                    calculator.add_event(event)

        # Compute metrics
        metrics = calculator.compute_metrics()

        metrics_dict = {
            'timestamp': timestamp,
            'bid_arrival_rate': metrics.bid_arrival_rate,
            'ask_arrival_rate': metrics.ask_arrival_rate,
            'total_arrival_rate': metrics.total_arrival_rate,
            'bid_cancel_ratio': metrics.bid_cancel_ratio,
            'ask_cancel_ratio': metrics.ask_cancel_ratio,
            'total_cancel_ratio': metrics.total_cancel_ratio,
            'order_book_intensity': metrics.order_book_intensity,
            'avg_queue_depth_bid': metrics.avg_queue_depth_bid,
            'avg_queue_depth_ask': metrics.avg_queue_depth_ask,
            'avg_time_between_events': metrics.avg_time_between_events,
        }

        metrics_list.append(metrics_dict)
        prev_snapshot = snapshot

    return pd.DataFrame(metrics_list)


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic order book snapshots
    np.random.seed(42)

    snapshots = []
    timestamp = 0
    bid_vol = 100.0
    ask_vol = 100.0
    mid_price = 50000.0

    for i in range(500):
        timestamp += np.random.exponential(0.1)  # Random time intervals

        # Random volume changes
        bid_vol += np.random.normal(0, 5)
        ask_vol += np.random.normal(0, 5)
        bid_vol = max(10, bid_vol)  # Keep positive
        ask_vol = max(10, ask_vol)

        # Price evolution
        mid_price += np.random.normal(0, 1)

        snapshot = {
            'timestamp': timestamp,
            'bids': [[mid_price - 0.5, bid_vol], [mid_price - 1, bid_vol * 0.8]],
            'asks': [[mid_price + 0.5, ask_vol], [mid_price + 1, ask_vol * 0.8]]
        }

        snapshots.append(snapshot)

    # Compute queue metrics
    print("Computing queue dynamics metrics...")
    metrics_df = compute_queue_metrics_from_snapshots(snapshots, window_size=50)

    print("\n" + "="*80)
    print("Queue Dynamics Metrics (first 10 rows)")
    print("="*80)

    display_cols = [
        'timestamp', 'bid_arrival_rate', 'ask_arrival_rate',
        'total_cancel_ratio', 'order_book_intensity'
    ]
    print(metrics_df[display_cols].head(10))

    print("\n" + "="*80)
    print("Queue Metrics Statistics")
    print("="*80)
    print(metrics_df[display_cols[1:]].describe())

    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Arrival rates
        axes[0].plot(metrics_df['timestamp'], metrics_df['bid_arrival_rate'],
                    label='Bid Arrival Rate', alpha=0.7)
        axes[0].plot(metrics_df['timestamp'], metrics_df['ask_arrival_rate'],
                    label='Ask Arrival Rate', alpha=0.7)
        axes[0].set_title('Order Arrival Rates')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Rate (events/sec)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Cancellation ratio
        axes[1].plot(metrics_df['timestamp'], metrics_df['total_cancel_ratio'],
                    label='Cancellation Ratio', alpha=0.7, color='red')
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='50% threshold')
        axes[1].set_title('Order Cancellation Ratio')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Cancellation Ratio')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim([0, 1])

        # Order book intensity
        axes[2].plot(metrics_df['timestamp'], metrics_df['order_book_intensity'],
                    label='Order Book Intensity', alpha=0.7, color='green')
        axes[2].set_title('Order Book Event Intensity')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Intensity (events/sec)')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('data/simulations/queue_dynamics_example.png', dpi=150)
        print("\nVisualization saved to: data/simulations/queue_dynamics_example.png")

    except ImportError:
        print("\nMatplotlib not available for visualization")
