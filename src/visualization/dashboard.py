"""
Streamlit Dashboard for HFT Order Book Forecasting

Interactive dashboard featuring:
- Real-time order book visualization
- Model predictions and performance
- Feature analysis and correlations
- PnL tracking and metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import os

# Page config
st.set_page_config(
    page_title="HFT Order Book Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("📈 High-Frequency Order Book Imbalance Forecasting")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Data source selection
    exchange = st.selectbox("Exchange", ["Binance", "Coinbase", "LOBSTER (NASDAQ)"])

    symbol = st.selectbox(
        "Symbol",
        (
            ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            if exchange == "Binance"
            else (
                ["BTC-USD", "ETH-USD"]
                if exchange == "Coinbase"
                else ["AAPL", "MSFT", "GOOGL"]
            )
        ),
    )

    # Model selection
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Model", ["LSTM", "Attention LSTM", "Transformer", "Ensemble"]
    )

    # Update frequency
    refresh_rate = st.slider("Refresh Rate (seconds)", 1, 60, 5)

    # Connect to API
    api_url = os.getenv("API_URL", "http://localhost:8000")
    st.markdown(f"**API:** {api_url}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Order Book", "🤖 Predictions", "📈 Performance", "🔍 Features"]
)

# Tab 1: Order Book Visualization
with tab1:
    st.header("Real-Time Order Book")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Order Book Heatmap")

        # Generate sample order book data (in production, fetch from DB/API)
        def generate_sample_orderbook():
            mid_price = 50000 + np.random.normal(0, 100)
            levels = 20

            bids = []
            asks = []

            for i in range(levels):
                bid_price = mid_price - (i + 1) * 10
                ask_price = mid_price + (i + 1) * 10
                bid_vol = np.random.uniform(0.5, 5.0)
                ask_vol = np.random.uniform(0.5, 5.0)

                bids.append([bid_price, bid_vol])
                asks.append([ask_price, ask_vol])

            return bids, asks, mid_price

        bids, asks, mid_price = generate_sample_orderbook()

        # Create order book visualization
        fig = go.Figure()

        # Bid side (green)
        bid_prices = [b[0] for b in bids]
        bid_vols = [b[1] for b in bids]

        fig.add_trace(
            go.Bar(
                x=bid_vols,
                y=bid_prices,
                orientation="h",
                name="Bids",
                marker=dict(color="green", opacity=0.7),
            )
        )

        # Ask side (red)
        ask_prices = [a[0] for a in asks]
        ask_vols = [-a[1] for a in asks]  # Negative for left side

        fig.add_trace(
            go.Bar(
                x=ask_vols,
                y=ask_prices,
                orientation="h",
                name="Asks",
                marker=dict(color="red", opacity=0.7),
            )
        )

        fig.update_layout(
            title="Order Book Depth",
            xaxis_title="Volume",
            yaxis_title="Price",
            height=500,
            barmode="relative",
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Metrics")

        # Display metrics
        spread = asks[0][0] - bids[0][0]
        spread_bps = (spread / mid_price) * 10000

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mid Price", f"${mid_price:,.2f}")
        col_b.metric("Spread", f"${spread:.2f}")
        col_c.metric("Spread (bps)", f"{spread_bps:.2f}")

        # Volume imbalance
        total_bid_vol = sum(b[1] for b in bids)
        total_ask_vol = sum(a[1] for a in asks)
        vol_imbalance = (total_bid_vol - total_ask_vol) / (
            total_bid_vol + total_ask_vol
        )

        st.metric("Volume Imbalance", f"{vol_imbalance:.3f}")

        # Time series of mid-price
        st.subheader("Price History")

        # Generate sample price history
        timestamps = pd.date_range(end=datetime.now(), periods=100, freq="1S")
        prices = mid_price + np.cumsum(np.random.normal(0, 5, 100))

        price_df = pd.DataFrame({"timestamp": timestamps, "price": prices})

        fig_price = px.line(
            price_df, x="timestamp", y="price", title="Mid-Price Evolution"
        )
        fig_price.update_layout(height=300)

        st.plotly_chart(fig_price, use_container_width=True)

# Tab 2: Predictions
with tab2:
    st.header("Model Predictions")

    col1, col2, col3 = st.columns(3)

    # Sample prediction (in production, fetch from API)
    prediction = np.random.choice(["UP", "DOWN", "FLAT"])
    confidence = np.random.uniform(0.5, 0.95)

    probs = {
        "UP": 0.4 if prediction != "UP" else confidence,
        "FLAT": 0.3 if prediction != "FLAT" else confidence,
        "DOWN": 0.3 if prediction != "DOWN" else confidence,
    }

    # Normalize
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    with col1:
        st.metric("Prediction", prediction)

    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    with col3:
        latency = np.random.uniform(10, 50)
        st.metric("Latency", f"{latency:.1f}ms")

    # Probability distribution
    st.subheader("Prediction Probabilities")

    fig_probs = go.Figure(
        data=[
            go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=[
                    "green" if k == "UP" else "gray" if k == "FLAT" else "red"
                    for k in probs.keys()
                ],
            )
        ]
    )
    fig_probs.update_layout(yaxis_title="Probability", height=300)

    st.plotly_chart(fig_probs, use_container_width=True)

    # Prediction history
    st.subheader("Prediction History")

    # Generate sample prediction history
    pred_history = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=datetime.now(), periods=50, freq="5S"),
            "prediction": np.random.choice(["UP", "DOWN", "FLAT"], 50),
            "confidence": np.random.uniform(0.5, 0.95, 50),
            "actual": np.random.choice(["UP", "DOWN", "FLAT"], 50),
        }
    )

    st.dataframe(pred_history.tail(10), use_container_width=True)

# Tab 3: Performance
with tab3:
    st.header("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Metrics")

        # Sample metrics
        accuracy = np.random.uniform(0.55, 0.70)
        precision = np.random.uniform(0.50, 0.65)
        recall = np.random.uniform(0.50, 0.65)
        f1 = 2 * (precision * recall) / (precision + recall)

        metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Value": [accuracy, precision, recall, f1],
            }
        )

        fig_metrics = px.bar(
            metrics_df, x="Metric", y="Value", title="Classification Performance"
        )
        fig_metrics.update_layout(yaxis_range=[0, 1])

        st.plotly_chart(fig_metrics, use_container_width=True)

    with col2:
        st.subheader("Economic Metrics")

        # Sample economic metrics
        sharpe = np.random.uniform(1.2, 2.5)
        max_dd = np.random.uniform(-0.15, -0.05)
        win_rate = np.random.uniform(0.50, 0.65)

        econ_df = pd.DataFrame(
            {
                "Metric": ["Sharpe Ratio", "Max Drawdown", "Win Rate"],
                "Value": [sharpe, max_dd, win_rate],
            }
        )

        st.dataframe(econ_df, use_container_width=True)

    # PnL Curve
    st.subheader("Cumulative PnL")

    # Generate sample PnL
    returns = np.random.normal(0.0005, 0.01, 1000)
    cum_returns = np.cumprod(1 + returns)

    pnl_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=datetime.now(), periods=1000, freq="1min"),
            "pnl": cum_returns,
        }
    )

    fig_pnl = px.line(pnl_df, x="timestamp", y="pnl", title="Cumulative Returns")
    st.plotly_chart(fig_pnl, use_container_width=True)

# Tab 4: Features
with tab4:
    st.header("Feature Analysis")

    # Feature importance
    st.subheader("Feature Importance")

    features = [
        "OFI_L1",
        "OFI_L5",
        "OFI_L10",
        "Micro_Price_Dev",
        "Volume_Imbalance",
        "Spread_BPS",
        "Realized_Vol_20",
        "Depth_Imbalance",
        "Liquidity_Conc",
    ]
    importance = np.random.uniform(0, 1, len(features))
    importance = importance / importance.sum()

    feat_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(
        "Importance", ascending=False
    )

    fig_feat = px.bar(
        feat_df, x="Feature", y="Importance", title="Feature Importance (SHAP Values)"
    )

    st.plotly_chart(fig_feat, use_container_width=True)

    # Feature correlation
    st.subheader("Feature Correlation Matrix")

    # Generate sample correlation matrix
    n_features = 8
    corr_matrix = np.random.uniform(-0.3, 0.3, (n_features, n_features))
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric

    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        x=features[:n_features],
        y=features[:n_features],
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )

    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    **HFT Order Book Imbalance Forecasting** | Built with Streamlit, PyTorch, FastAPI
    | [GitHub](https://github.com/mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine)
"""
)

# Auto-refresh (optional)
if st.button("🔄 Refresh Data"):
    st.experimental_rerun()
