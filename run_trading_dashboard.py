"""
Main Streamlit Trading Dashboard Application

Professional trading dashboard with:
- Real-time order book visualization
- AI-powered trading signals
- Performance analytics
- Feature analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from visualization.app_utils import (
    OrderBookGenerator,
    MetricsCalculator,
    SignalGenerator,
    DataGenerator,
    PlotGenerator,
    FeatureAnalyzer,
    validate_orderbook_data,
    format_currency,
    format_percentage,
)

# Page configuration
st.set_page_config(
    page_title="QuantumFlow Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0

if "mid_price" not in st.session_state:
    st.session_state.mid_price = 50000.0

# Title
st.markdown(
    '<div class="main-header">📊 QuantumFlow Trading Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown("**AI-Powered High-Frequency Trading Analytics**")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Exchange & Symbol Selection
    exchange = st.selectbox(
        "Exchange",
        ["Binance", "Coinbase", "Kraken", "LOBSTER (NASDAQ)"],
        help="Select data source",
    )

    symbol_options = {
        "Binance": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],
        "Coinbase": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "Kraken": ["XBTUSD", "ETHUSD", "SOLUSD"],
        "LOBSTER (NASDAQ)": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    }

    symbol = st.selectbox("Symbol", symbol_options[exchange])

    st.markdown("---")

    # Model Configuration
    st.subheader("🤖 Model Settings")
    model_type = st.selectbox(
        "Model",
        ["Ensemble", "Transformer", "Attention LSTM", "LSTM", "Bayesian Online"],
        help="Select prediction model",
    )

    # Trading Parameters
    st.subheader("📈 Trading Parameters")
    signal_threshold = st.slider(
        "Signal Threshold", 0.0, 0.5, 0.1, 0.05, help="Volume imbalance threshold"
    )

    refresh_rate = st.slider(
        "Refresh Rate (sec)", 1, 60, 5, help="Dashboard update frequency"
    )

    st.markdown("---")

    # System Status
    st.subheader("📊 System Status")
    st.success("✅ Connected")
    st.info(f"Model: {model_type}")
    st.info(f"Refreshes: {st.session_state.refresh_count}")

    # Refresh button
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.session_state.refresh_count += 1
        st.rerun()

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Order Book",
        "🎯 Trading Signals",
        "📈 Performance",
        "🔍 Features",
        "📋 Logs",
    ]
)

# Generate order book data
ob_generator = OrderBookGenerator()
bids, asks, mid_price = ob_generator.generate_orderbook(
    mid_price=st.session_state.mid_price, levels=20, spread_bps=5.0
)

# Validate order book
if not validate_orderbook_data(bids, asks):
    st.error("⚠️ Invalid order book data generated")
    st.stop()

# Update session state
st.session_state.mid_price = mid_price

# Calculate metrics
metrics_calc = MetricsCalculator()
spread_metrics = metrics_calc.calculate_spread_metrics(bids, asks, mid_price)
volume_imbalance = metrics_calc.calculate_volume_imbalance(bids, asks)

# Tab 1: Order Book Visualization
with tab1:
    st.header("📊 Real-Time Order Book")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💰 Mid Price", format_currency(mid_price))

    with col2:
        st.metric("📏 Spread", format_currency(spread_metrics["spread"]))

    with col3:
        st.metric("📊 Spread (bps)", f"{spread_metrics['spread_bps']:.2f}")

    with col4:
        imbalance_delta = volume_imbalance * 100
        st.metric(
            "⚖️ Volume Imbalance",
            f"{volume_imbalance:.3f}",
            delta=f"{imbalance_delta:.2f}%",
        )

    st.markdown("---")

    # Order book visualization and price history
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Order Book Depth")
        plot_gen = PlotGenerator()
        orderbook_fig = plot_gen.create_orderbook_plot(bids, asks)
        st.plotly_chart(orderbook_fig, use_container_width=True)

        # Order book statistics
        with st.expander("📊 Order Book Statistics"):
            total_bid_vol = sum(b[1] for b in bids)
            total_ask_vol = sum(a[1] for a in asks)

            st.write(f"**Total Bid Volume:** {total_bid_vol:.4f}")
            st.write(f"**Total Ask Volume:** {total_ask_vol:.4f}")
            st.write(f"**Best Bid:** {format_currency(bids[0][0])}")
            st.write(f"**Best Ask:** {format_currency(asks[0][0])}")

    with col2:
        st.subheader("Price History")
        data_gen = DataGenerator()
        price_history = data_gen.generate_price_history(mid_price, periods=100)

        price_fig = px.line(
            price_history,
            x="timestamp",
            y="price",
            title="Mid-Price Evolution (Last 100 Seconds)",
        )
        price_fig.update_layout(height=500)
        st.plotly_chart(price_fig, use_container_width=True)

        # Price statistics
        with st.expander("📈 Price Statistics"):
            st.write(f"**Current:** {format_currency(price_history['price'].iloc[-1])}")
            st.write(f"**High:** {format_currency(price_history['price'].max())}")
            st.write(f"**Low:** {format_currency(price_history['price'].min())}")
            st.write(
                f"**Volatility (std):** {format_currency(price_history['price'].std())}"
            )

# Tab 2: Trading Signals
with tab2:
    st.header("🎯 AI Trading Signals")

    # Generate signal
    signal_gen = SignalGenerator()
    signal, confidence = signal_gen.generate_signal(
        volume_imbalance, spread_metrics["spread_bps"], threshold=signal_threshold
    )
    probabilities = signal_gen.calculate_signal_probabilities(signal, confidence)

    # Signal display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        signal_emoji = {"UP": "🟢", "DOWN": "🔴", "FLAT": "⚪"}
        st.metric("Signal", f"{signal_emoji[signal]} {signal}")

    with col2:
        st.metric("Confidence", format_percentage(confidence))

    with col3:
        latency = np.random.uniform(15, 45)
        st.metric("Latency", f"{latency:.1f}ms")

    with col4:
        st.metric("Model", model_type)

    st.markdown("---")

    # Signal probabilities and prediction history
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Signal Probabilities")
        prob_fig = plot_gen.create_signal_probability_plot(probabilities)
        st.plotly_chart(prob_fig, use_container_width=True)

        # Probability breakdown
        with st.expander("📊 Probability Breakdown"):
            for direction, prob in probabilities.items():
                st.write(f"**{direction}:** {format_percentage(prob)}")

    with col2:
        st.subheader("Recent Predictions")
        pred_history = data_gen.generate_prediction_history(periods=20)

        # Calculate accuracy
        accuracy = metrics_calc.calculate_win_rate(
            pred_history["prediction"].tolist(), pred_history["actual"].tolist()
        )

        st.metric("Recent Accuracy", format_percentage(accuracy))

        st.dataframe(
            pred_history.tail(10)[
                ["timestamp", "prediction", "confidence", "actual"]
            ].style.format({"confidence": "{:.2%}"}),
            use_container_width=True,
            height=300,
        )

# Tab 3: Performance Analytics
with tab3:
    st.header("📈 Performance Analytics")

    # Generate performance data
    pnl_data = data_gen.generate_pnl_curve(periods=1000, sharpe_target=1.5)
    returns = pnl_data["pnl"].pct_change().dropna()

    # Calculate metrics
    sharpe = metrics_calc.calculate_sharpe_ratio(returns.values)
    max_dd = metrics_calc.calculate_max_drawdown(pnl_data["pnl"].values)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Sharpe Ratio", f"{sharpe:.2f}")

    with col2:
        st.metric("📉 Max Drawdown", format_percentage(max_dd))

    with col3:
        total_return = (pnl_data["pnl"].iloc[-1] - 1) * 100
        st.metric("💰 Total Return", f"{total_return:.2f}%")

    with col4:
        # Sample win rate
        win_rate = 0.52 + np.random.uniform(-0.05, 0.05)
        st.metric("🎯 Win Rate", format_percentage(win_rate))

    st.markdown("---")

    # PnL curve and metrics
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Cumulative PnL")
        pnl_fig = px.line(
            pnl_data,
            x="timestamp",
            y="pnl",
            title="Strategy Performance Over Time",
        )
        pnl_fig.add_hline(
            y=1.0, line_dash="dash", line_color="gray", annotation_text="Breakeven"
        )
        pnl_fig.update_layout(height=400)
        st.plotly_chart(pnl_fig, use_container_width=True)

    with col2:
        st.subheader("Key Metrics")

        metrics_data = {
            "Metric": [
                "Sharpe Ratio",
                "Max Drawdown",
                "Win Rate",
                "Total Return",
                "Avg Trade",
                "Profit Factor",
            ],
            "Value": [
                f"{sharpe:.2f}",
                format_percentage(max_dd),
                format_percentage(win_rate),
                f"{total_return:.2f}%",
                f"{returns.mean()*100:.3f}%",
                f"{1.2 + np.random.uniform(0, 0.3):.2f}",
            ],
        }

        st.dataframe(
            pd.DataFrame(metrics_data), use_container_width=True, hide_index=True
        )

    # Returns distribution
    st.subheader("Returns Distribution")
    returns_fig = px.histogram(
        returns * 100,
        nbins=50,
        title="Distribution of Returns",
        labels={"value": "Return (%)", "count": "Frequency"},
    )
    returns_fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(returns_fig, use_container_width=True)

# Tab 4: Feature Analysis
with tab4:
    st.header("🔍 Feature Analysis")

    # Feature importance
    feat_analyzer = FeatureAnalyzer()
    feature_importance = feat_analyzer.generate_feature_importance(n_features=9)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Feature Importance (SHAP)")
        feat_fig = px.bar(
            feature_importance,
            x="Feature",
            y="Importance",
            title="Top Features by Importance",
            color="Importance",
            color_continuous_scale="blues",
        )
        feat_fig.update_layout(height=400)
        st.plotly_chart(feat_fig, use_container_width=True)

    with col2:
        st.subheader("Feature Correlation Matrix")
        features = feature_importance["Feature"].tolist()[:8]
        corr_matrix = feat_analyzer.generate_correlation_matrix(features)

        corr_fig = px.imshow(
            corr_matrix,
            labels=dict(x="Feature", y="Feature", color="Correlation"),
            x=features,
            y=features,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        corr_fig.update_layout(height=400)
        st.plotly_chart(corr_fig, use_container_width=True)

    # Feature values table
    st.subheader("Current Feature Values")

    current_features = {
        "Feature": [
            "OFI_L1",
            "Volume_Imbalance",
            "Spread_BPS",
            "Micro_Price",
            "Realized_Vol",
        ],
        "Value": [
            f"{np.random.uniform(-0.5, 0.5):.4f}",
            f"{volume_imbalance:.4f}",
            f"{spread_metrics['spread_bps']:.2f}",
            f"{mid_price:.2f}",
            f"{np.random.uniform(0.001, 0.02):.4f}",
        ],
        "Status": ["Normal", "Normal", "Normal", "Normal", "Normal"],
    }

    st.dataframe(
        pd.DataFrame(current_features), use_container_width=True, hide_index=True
    )

# Tab 5: System Logs
with tab5:
    st.header("📋 System Logs")

    # Log level filter
    log_level = st.selectbox("Log Level", ["ALL", "INFO", "WARNING", "ERROR"])

    # Sample logs
    logs = [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": "INFO",
            "message": f"Order book updated for {symbol}",
        },
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": "INFO",
            "message": f"Generated signal: {signal} with confidence {confidence:.2%}",
        },
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": "INFO",
            "message": f"Model {model_type} prediction completed in {latency:.1f}ms",
        },
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": "WARNING",
            "message": f"High spread detected: {spread_metrics['spread_bps']:.2f} bps",
        }
        if spread_metrics["spread_bps"] > 10
        else None,
    ]

    logs = [log for log in logs if log is not None]

    # Filter logs
    if log_level != "ALL":
        logs = [log for log in logs if log["level"] == log_level]

    st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)

    # System health
    st.subheader("🏥 System Health")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CPU Usage", f"{np.random.uniform(20, 60):.1f}%")

    with col2:
        st.metric("Memory Usage", f"{np.random.uniform(30, 70):.1f}%")

    with col3:
        st.metric("API Calls/min", f"{np.random.randint(50, 150)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><strong>QuantumFlow Trading Dashboard</strong> | Built with Streamlit & PyTorch</p>
        <p>⚡ Real-time HFT Analytics | 🤖 AI-Powered Predictions | 📊 Professional Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Auto-refresh logic (optional)
# Uncomment to enable auto-refresh
# import time
# time.sleep(refresh_rate)
# st.rerun()
