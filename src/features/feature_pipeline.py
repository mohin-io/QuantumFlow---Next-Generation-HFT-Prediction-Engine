"""
Feature Pipeline - Integrated Feature Engineering

Combines all microstructure features into a unified pipeline:
- Order Flow Imbalance (OFI)
- Micro-price
- Volume Profiles
- Queue Dynamics
- Realized Volatility

Provides a single interface for extracting all features from order book data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict

from features.order_flow_imbalance import (
    OrderFlowImbalanceCalculator,
    OrderBookState,
    compute_ofi_from_dataframe
)
from features.micro_price import (
    MicroPriceCalculator,
    compute_micro_price_features
)
from features.volume_profiles import (
    VolumeProfileCalculator,
    compute_volume_features
)
from features.queue_dynamics import (
    compute_queue_metrics_from_snapshots
)
from features.realized_volatility import (
    compute_ohlc_from_snapshots,
    compute_volatility_features
)


@dataclass
class FeaturePipelineConfig:
    """Configuration for feature pipeline."""

    # OFI settings
    ofi_levels: List[int] = None
    ofi_windows: List[int] = None

    # Micro-price settings
    micro_price_depth: int = 3
    use_adaptive_fair_value: bool = True
    adaptive_alpha: float = 0.3

    # Volume profile settings
    volume_depth_levels: int = 20

    # Queue dynamics settings
    queue_window_size: int = 100

    # Volatility settings
    volatility_windows: List[int] = None
    ohlc_bar_size: int = 20

    def __post_init__(self):
        """Set defaults for list fields."""
        if self.ofi_levels is None:
            self.ofi_levels = [1, 5, 10]
        if self.ofi_windows is None:
            self.ofi_windows = [10, 50, 100]
        if self.volatility_windows is None:
            self.volatility_windows = [20, 50, 100]


class FeaturePipeline:
    """
    Unified feature engineering pipeline.

    Orchestrates all feature calculators to produce a complete
    feature set from order book snapshots.
    """

    def __init__(self, config: Optional[FeaturePipelineConfig] = None):
        """
        Initialize feature pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or FeaturePipelineConfig()

    def compute_all_features(
        self,
        snapshots: Union[pd.DataFrame, List[Dict]],
        include_volatility: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features from order book snapshots.

        Args:
            snapshots: Order book snapshots (DataFrame or list of dicts)
            include_volatility: Whether to compute volatility features

        Returns:
            DataFrame with all features
        """
        # Convert to DataFrame if needed
        if isinstance(snapshots, list):
            df = pd.DataFrame(snapshots)
        else:
            df = snapshots.copy()

        print("Computing Order Flow Imbalance...")
        df = compute_ofi_from_dataframe(
            df,
            levels=self.config.ofi_levels,
            window_sizes=self.config.ofi_windows
        )

        print("Computing Micro-price features...")
        df = compute_micro_price_features(
            df,
            depth_levels=self.config.micro_price_depth,
            use_adaptive=self.config.use_adaptive_fair_value,
            alpha=self.config.adaptive_alpha
        )

        print("Computing Volume Profile features...")
        df = compute_volume_features(
            df,
            depth_levels=self.config.volume_depth_levels
        )

        print("Computing Queue Dynamics...")
        queue_metrics = compute_queue_metrics_from_snapshots(
            df.to_dict('records'),
            window_size=self.config.queue_window_size
        )

        # Merge queue metrics
        for col in queue_metrics.columns:
            if col != 'timestamp':
                df[col] = queue_metrics[col].values

        # Volatility features (optional, computationally expensive)
        if include_volatility:
            print("Computing Realized Volatility...")
            ohlc_df = compute_ohlc_from_snapshots(
                df.to_dict('records'),
                window_ticks=self.config.ohlc_bar_size
            )

            vol_df = compute_volatility_features(
                ohlc_df,
                windows=self.config.volatility_windows
            )

            # Merge volatility features (forward fill for alignment)
            # This is simplified - in production you'd do proper time alignment
            for col in vol_df.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close']:
                    # Repeat values to match snapshot frequency
                    repeated_values = np.repeat(
                        vol_df[col].values,
                        self.config.ohlc_bar_size
                    )
                    # Pad or truncate to match df length
                    if len(repeated_values) < len(df):
                        repeated_values = np.pad(
                            repeated_values,
                            (0, len(df) - len(repeated_values)),
                            mode='edge'
                        )
                    else:
                        repeated_values = repeated_values[:len(df)]

                    df[col] = repeated_values

        print(f"Feature engineering complete! Total features: {len(df.columns)}")

        return df

    def get_feature_names(self, include_volatility: bool = True) -> List[str]:
        """
        Get list of all feature names that will be generated.

        Args:
            include_volatility: Whether volatility features are included

        Returns:
            List of feature names
        """
        features = []

        # OFI features
        for level in self.config.ofi_levels:
            features.append(f'ofi_L{level}')
            for window in self.config.ofi_windows:
                features.extend([
                    f'ofi_L{level}_mean_{window}',
                    f'ofi_L{level}_std_{window}',
                    f'ofi_L{level}_sum_{window}'
                ])

        # Micro-price features
        features.extend([
            'micro_price',
            'mid_price',
            'weighted_mid_price',
            'micro_price_deviation',
            'micro_price_bps_deviation'
        ])

        if self.config.use_adaptive_fair_value:
            features.extend([
                'adaptive_fair_value',
                'price_vs_fair_value',
                'price_vs_fair_value_bps'
            ])

        # Volume features
        features.extend([
            'total_bid_volume',
            'total_ask_volume',
            'volume_imbalance',
            'volume_imbalance_ratio',
            'depth_imbalance',
            'liquidity_concentration_bid',
            'liquidity_concentration_ask',
            'spread',
            'spread_bps',
            'relative_spread'
        ])

        # Queue dynamics
        features.extend([
            'bid_arrival_rate',
            'ask_arrival_rate',
            'total_arrival_rate',
            'bid_cancel_ratio',
            'ask_cancel_ratio',
            'total_cancel_ratio',
            'order_book_intensity',
            'avg_queue_depth_bid',
            'avg_queue_depth_ask',
            'avg_time_between_events'
        ])

        # Volatility features
        if include_volatility:
            for window in self.config.volatility_windows:
                features.extend([
                    f'rv_{window}',
                    f'parkinson_vol_{window}',
                    f'garman_klass_vol_{window}',
                    f'rogers_satchell_vol_{window}'
                ])

        return features

    def compute_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for all features.

        Args:
            df: DataFrame with computed features

        Returns:
            DataFrame with feature statistics
        """
        feature_names = [col for col in df.columns if col not in ['timestamp', 'exchange', 'symbol', 'bids', 'asks']]

        stats = df[feature_names].describe().T
        stats['missing_pct'] = (df[feature_names].isna().sum() / len(df) * 100).values

        return stats


def create_training_dataset(
    features_df: pd.DataFrame,
    prediction_horizon: int = 50,
    threshold_bps: float = 5.0,
    sequence_length: int = 100
) -> Dict[str, np.ndarray]:
    """
    Create training dataset with sequences and labels.

    Args:
        features_df: DataFrame with computed features
        prediction_horizon: How many ticks ahead to predict
        threshold_bps: Threshold in basis points for flat classification
        sequence_length: Length of input sequences

    Returns:
        Dictionary with X (features) and y (labels)
    """
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in features_df.columns
                   if col not in ['timestamp', 'exchange', 'symbol', 'bids', 'asks']]

    # Ensure we have mid_price
    if 'mid_price' not in features_df.columns:
        raise ValueError("mid_price column required for label generation")

    # Compute future returns
    features_df['future_mid_price'] = features_df['mid_price'].shift(-prediction_horizon)
    features_df['future_return_bps'] = (
        (features_df['future_mid_price'] - features_df['mid_price']) /
        features_df['mid_price'] * 10000
    )

    # Create labels (0: down, 1: flat, 2: up)
    def classify_movement(return_bps):
        if pd.isna(return_bps):
            return np.nan
        elif return_bps > threshold_bps:
            return 2  # up
        elif return_bps < -threshold_bps:
            return 0  # down
        else:
            return 1  # flat

    features_df['label'] = features_df['future_return_bps'].apply(classify_movement)

    # Remove rows with NaN
    features_df = features_df.dropna()

    # Create sequences
    X_list = []
    y_list = []

    for i in range(len(features_df) - sequence_length):
        sequence = features_df[feature_cols].iloc[i:i+sequence_length].values
        label = features_df['label'].iloc[i+sequence_length]

        if not np.isnan(label):
            X_list.append(sequence)
            y_list.append(int(label))

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Created dataset: X shape {X.shape}, y shape {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    return {
        'X': X,
        'y': y,
        'feature_names': feature_cols,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon
    }


# Example usage
if __name__ == "__main__":
    # Generate synthetic order book data
    np.random.seed(42)

    snapshots = []
    mid_price = 50000.0

    for i in range(500):
        mid_price += np.random.normal(0, 10)

        bids = []
        asks = []

        for j in range(20):
            bid_price = mid_price - (j + 1) * 0.5
            ask_price = mid_price + (j + 1) * 0.5
            bid_vol = np.random.uniform(10, 100)
            ask_vol = np.random.uniform(10, 100)

            bids.append([bid_price, bid_vol])
            asks.append([ask_price, ask_vol])

        snapshots.append({
            'timestamp': i,
            'exchange': 'binance',
            'symbol': 'BTCUSDT',
            'bids': bids,
            'asks': asks
        })

    # Create pipeline
    config = FeaturePipelineConfig(
        ofi_levels=[1, 5],
        ofi_windows=[10, 50],
        volatility_windows=[20],
        ohlc_bar_size=10
    )

    pipeline = FeaturePipeline(config)

    # Compute features
    print("="*80)
    print("Feature Pipeline Example")
    print("="*80)

    features_df = pipeline.compute_all_features(snapshots, include_volatility=True)

    print("\n" + "="*80)
    print("Sample Features (first 5 rows, selected columns)")
    print("="*80)

    sample_cols = ['mid_price', 'ofi_L1', 'micro_price_bps_deviation',
                   'volume_imbalance_ratio', 'spread_bps']
    print(features_df[sample_cols].head())

    print("\n" + "="*80)
    print("Feature Statistics")
    print("="*80)

    stats = pipeline.compute_feature_statistics(features_df)
    print(stats.head(20))

    print("\n" + "="*80)
    print("Creating Training Dataset")
    print("="*80)

    dataset = create_training_dataset(
        features_df,
        prediction_horizon=10,
        threshold_bps=5.0,
        sequence_length=50
    )

    print(f"\nDataset ready for training:")
    print(f"  Input shape: {dataset['X'].shape}")
    print(f"  Labels shape: {dataset['y'].shape}")
    print(f"  Number of features: {len(dataset['feature_names'])}")
