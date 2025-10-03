"""
Real-Time HFT Trading Dashboard with Live Market Data

Features:
1. Live Order Book Visualization (multi-exchange)
2. Real-Time Signal Generation
3. Execution Simulation with Live Prices
4. Performance Tracking
5. Arbitrage Detection
6. Market Microstructure Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import time
import asyncio

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.live_data_connector import (
    LiveDataAggregator,
    BinanceConnector,
    CoinbaseConnector,
)
from features.order_flow_imbalance import OFICalculator, OrderBookState
from features.micro_price import MicroPriceCalculator

# Page config
st.set_page_config(
    page_title="HFT Live Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-positive {
        color: #00C853;
        font-weight: bold;
    }
    .metric-negative {
        color: #D32F2F;
        font-weight: bold;
    }
    .signal-buy {
        background: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .signal-sell {
        background: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .signal-neutral {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #4CAF50;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "aggregator" not in st.session_state:
    st.session_state.aggregator = LiveDataAggregator()
    st.session_state.last_update = datetime.now()
    st.session_state.order_book_history = []
    st.session_state.trade_history = []
    st.session_state.signals = []
    st.session_state.pnl = 0.0
    st.session_state.trades_executed = 0


def fetch_live_data(exchange: str, symbol: str):
    """Fetch live data from exchange."""
    try:
        if exchange == "Binance":
            book = st.session_state.aggregator.binance.get_order_book(symbol, limit=20)
            stats = st.session_state.aggregator.binance.get_24h_stats(symbol)
            return book, stats
        elif exchange == "Coinbase":
            book = st.session_state.aggregator.coinbase.get_order_book(symbol, level=2)
            stats = {}
            return book, stats
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None


def create_order_book_heatmap(book):
    """Create order book depth heatmap."""
    if not book:
        return None

    # Prepare data
    bid_prices = [b[0] for b in book.bids[:20]]
    bid_sizes = [b[1] for b in book.bids[:20]]
    ask_prices = [a[0] for a in book.asks[:20]]
    ask_sizes = [a[1] for a in book.asks[:20]]

    # Create figure
    fig = go.Figure()

    # Bids (green)
    fig.add_trace(
        go.Bar(
            y=bid_prices,
            x=bid_sizes,
            orientation="h",
            name="Bids",
            marker=dict(color="green", opacity=0.7),
            text=[f"${p:,.2f}" for p in bid_prices],
            textposition="auto",
        )
    )

    # Asks (red)
    fig.add_trace(
        go.Bar(
            y=ask_prices,
            x=[-s for s in ask_sizes],  # Negative for left side
            orientation="h",
            name="Asks",
            marker=dict(color="red", opacity=0.7),
            text=[f"${p:,.2f}" for p in ask_prices],
            textposition="auto",
        )
    )

    # Calculate mid price
    mid_price = (bid_prices[0] + ask_prices[0]) / 2

    fig.add_hline(
        y=mid_price,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mid: ${mid_price:,.2f}",
    )

    fig.update_layout(
        title=f"Live Order Book - {book.exchange.upper()} {book.symbol}",
        xaxis_title="Size (Negative = Asks, Positive = Bids)",
        yaxis_title="Price ($)",
        barmode="overlay",
        height=600,
        showlegend=True,
    )

    return fig


def calculate_features(book):
    """Calculate microstructure features from order book."""
    if not book:
        return {}

    # Convert to OrderBookState
    state = OrderBookState(
        timestamp=book.timestamp, bids=book.bids[:10], asks=book.asks[:10]
    )

    # OFI Calculator
    ofi_calc = OFICalculator()
    ofi_calc.update(state)

    # Micro-price
    micro_calc = MicroPriceCalculator()
    micro_price = micro_calc.calculate(state)
    spread = micro_calc.calculate_spread(state)
    spread_bps = micro_calc.calculate_spread_bps(state)

    # Volume imbalance
    total_bid_vol = sum(b[1] for b in book.bids[:10])
    total_ask_vol = sum(a[1] for a in book.asks[:10])
    vol_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

    # Best bid/ask
    best_bid = book.bids[0][0] if book.bids else 0
    best_ask = book.asks[0][0] if book.asks else 0
    mid_price = (best_bid + best_ask) / 2

    return {
        "timestamp": datetime.fromtimestamp(book.timestamp / 1000),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "micro_price": micro_price,
        "spread": spread,
        "spread_bps": spread_bps,
        "volume_imbalance": vol_imbalance,
        "total_bid_volume": total_bid_vol,
        "total_ask_volume": total_ask_vol,
    }


def generate_trading_signal(features):
    """Generate trading signal from features."""
    if not features:
        return "NEUTRAL", 0.5

    # Simple rule-based signal
    vol_imb = features["volume_imbalance"]
    spread_bps = features["spread_bps"]

    # Strong buy signal
    if vol_imb > 0.15 and spread_bps < 10:
        signal = "BUY"
        confidence = min(0.95, 0.6 + abs(vol_imb) * 2)

    # Strong sell signal
    elif vol_imb < -0.15 and spread_bps < 10:
        signal = "SELL"
        confidence = min(0.95, 0.6 + abs(vol_imb) * 2)

    # Neutral
    else:
        signal = "NEUTRAL"
        confidence = 0.5 + abs(vol_imb)

    return signal, confidence


def simulate_execution(signal, confidence, price, position_size=0.01):
    """Simulate trade execution."""
    if confidence < 0.65:
        return None

    # Estimate execution price with slippage
    slippage_bps = 5  # 5 basis points
    if signal == "BUY":
        exec_price = price * (1 + slippage_bps / 10000)
        side = "BUY"
    elif signal == "SELL":
        exec_price = price * (1 - slippage_bps / 10000)
        side = "SELL"
    else:
        return None

    # Transaction cost
    fee_bps = 10  # 10 basis points
    fee = position_size * exec_price * fee_bps / 10000

    return {
        "timestamp": datetime.now(),
        "signal": signal,
        "confidence": confidence,
        "entry_price": exec_price,
        "size": position_size,
        "fee": fee,
        "side": side,
    }


def main():
    """Main dashboard."""

    # Header
    st.markdown(
        '<div class="main-header">📈 HFT Live Trading Dashboard <span class="live-indicator"></span></div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Exchange selection
        exchange = st.selectbox("Exchange", ["Binance", "Coinbase"], index=0)

        # Symbol selection
        if exchange == "Binance":
            default_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
            symbol = st.selectbox("Symbol", default_symbols, index=0)
        else:
            default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"]
            symbol = st.selectbox("Symbol", default_symbols, index=0)

        st.markdown("---")

        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 3)

        st.markdown("---")

        # Trading parameters
        st.markdown("### Trading Parameters")
        confidence_threshold = st.slider("Min Confidence", 0.5, 0.95, 0.65, 0.05)
        position_size = st.number_input("Position Size (BTC)", 0.001, 1.0, 0.01, 0.001)

        st.markdown("---")

        # Manual refresh
        if st.button("🔄 Refresh Now", type="primary"):
            st.rerun()

        st.markdown("---")
        st.markdown(
            f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}"
        )

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Order Book",
            "🎯 Trading Signals",
            "💰 Performance",
            "🔍 Arbitrage",
            "📈 Analytics",
        ]
    )

    # Fetch live data
    book, stats = fetch_live_data(exchange, symbol)

    if book:
        st.session_state.last_update = datetime.now()

        # Calculate features
        features = calculate_features(book)

        # Generate signal
        signal, confidence = generate_trading_signal(features)

        # TAB 1: Order Book Visualization
        with tab1:
            st.markdown("## Live Order Book")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Best Bid", f"${features['best_bid']:,.2f}", delta=None)

            with col2:
                st.metric("Best Ask", f"${features['best_ask']:,.2f}", delta=None)

            with col3:
                st.metric(
                    "Spread",
                    f"${features['spread']:.2f}",
                    delta=f"{features['spread_bps']:.1f} bps",
                )

            with col4:
                if stats:
                    change_pct = stats.get("price_change_pct", 0)
                    st.metric(
                        "24h Change",
                        f"{change_pct:+.2f}%",
                        delta=f"${stats.get('price_change', 0):+,.2f}",
                    )

            # Order book heatmap
            fig_book = create_order_book_heatmap(book)
            if fig_book:
                st.plotly_chart(fig_book, use_container_width=True)

            # Depth metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Bid Volume (10 levels)",
                    f"{features['total_bid_volume']:.4f}",
                    delta=None,
                )

            with col2:
                st.metric(
                    "Ask Volume (10 levels)",
                    f"{features['total_ask_volume']:.4f}",
                    delta=None,
                )

            with col3:
                vol_imb = features["volume_imbalance"]
                st.metric(
                    "Volume Imbalance",
                    f"{vol_imb:+.3f}",
                    delta="Bullish" if vol_imb > 0 else "Bearish",
                )

        # TAB 2: Trading Signals
        with tab2:
            st.markdown("## Real-Time Trading Signals")

            # Current signal
            signal_class = f"signal-{signal.lower()}"
            signal_emoji = (
                "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            )

            st.markdown(
                f"""
                <div class="{signal_class}">
                    <h3>{signal_emoji} {signal}</h3>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>Price:</strong> ${features['mid_price']:,.2f}</p>
                    <p><strong>Micro-Price:</strong> ${features['micro_price']:,.2f}</p>
                    <p><strong>Volume Imbalance:</strong> {features['volume_imbalance']:+.3f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Execute simulation
            if confidence >= confidence_threshold:
                st.success(
                    f"✅ Signal meets confidence threshold ({confidence:.1%} >= {confidence_threshold:.1%})"
                )

                if st.button(f"🚀 Simulate {signal} Order", type="primary"):
                    trade = simulate_execution(
                        signal, confidence, features["mid_price"], position_size
                    )
                    if trade:
                        st.session_state.trade_history.append(trade)
                        st.session_state.trades_executed += 1
                        st.success(
                            f"Trade simulated: {signal} {position_size} @ ${trade['entry_price']:,.2f}"
                        )
            else:
                st.warning(
                    f"⚠️ Signal below confidence threshold ({confidence:.1%} < {confidence_threshold:.1%})"
                )

            # Signal history
            st.markdown("### Signal History")

            if st.session_state.trade_history:
                history_df = pd.DataFrame(st.session_state.trade_history)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No trades executed yet. Adjust parameters and wait for signals."
                )

            # Feature details
            st.markdown("### Feature Details")

            feature_df = pd.DataFrame(
                [
                    {"Feature": "Mid Price", "Value": f"${features['mid_price']:,.2f}"},
                    {
                        "Feature": "Micro Price",
                        "Value": f"${features['micro_price']:,.2f}",
                    },
                    {
                        "Feature": "Spread (bps)",
                        "Value": f"{features['spread_bps']:.2f}",
                    },
                    {
                        "Feature": "Volume Imbalance",
                        "Value": f"{features['volume_imbalance']:+.4f}",
                    },
                ]
            )

            st.dataframe(feature_df, use_container_width=True, hide_index=True)

        # TAB 3: Performance Tracking
        with tab3:
            st.markdown("## Performance Tracking")

            col1, col2, col3, col4 = st.columns(4)

            # Calculate P&L (simulated)
            if st.session_state.trade_history:
                trades_df = pd.DataFrame(st.session_state.trade_history)

                # Simulate P&L based on current price
                total_pnl = 0
                for trade in st.session_state.trade_history:
                    if trade["side"] == "BUY":
                        pnl = (features["mid_price"] - trade["entry_price"]) * trade[
                            "size"
                        ]
                    else:
                        pnl = (trade["entry_price"] - features["mid_price"]) * trade[
                            "size"
                        ]

                    pnl -= trade["fee"]
                    total_pnl += pnl

                st.session_state.pnl = total_pnl

                with col1:
                    st.metric(
                        "Total PnL",
                        f"${total_pnl:,.2f}",
                        delta=f"{(total_pnl / (position_size * features['mid_price'])) * 100:+.2f}%",
                    )

                with col2:
                    st.metric("Trades Executed", st.session_state.trades_executed)

                with col3:
                    avg_pnl = total_pnl / len(st.session_state.trade_history)
                    st.metric("Avg PnL/Trade", f"${avg_pnl:,.2f}")

                with col4:
                    win_rate = len(
                        [
                            t
                            for t in st.session_state.trade_history
                            if (features["mid_price"] - t["entry_price"])
                            * (1 if t["side"] == "BUY" else -1)
                            > 0
                        ]
                    ) / len(st.session_state.trade_history)
                    st.metric("Win Rate", f"{win_rate:.1%}")

                # P&L chart
                st.markdown("### Cumulative P&L")

                cumulative_pnl = []
                running_pnl = 0
                for trade in st.session_state.trade_history:
                    if trade["side"] == "BUY":
                        pnl = (features["mid_price"] - trade["entry_price"]) * trade[
                            "size"
                        ]
                    else:
                        pnl = (trade["entry_price"] - features["mid_price"]) * trade[
                            "size"
                        ]
                    pnl -= trade["fee"]
                    running_pnl += pnl
                    cumulative_pnl.append(running_pnl)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        y=cumulative_pnl,
                        mode="lines+markers",
                        name="Cumulative PnL",
                        line=dict(color="green" if total_pnl > 0 else "red", width=2),
                    )
                )

                fig.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Trade Number",
                    yaxis_title="P&L ($)",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No performance data yet. Execute some trades to see results.")

        # TAB 4: Arbitrage Detection
        with tab4:
            st.markdown("## Cross-Exchange Arbitrage")

            symbol_map = {
                "binance": symbol if exchange == "Binance" else "BTCUSDT",
                "coinbase": symbol if exchange == "Coinbase" else "BTC-USD",
            }

            opportunities = (
                st.session_state.aggregator.calculate_arbitrage_opportunities(
                    symbol_map
                )
            )

            if opportunities:
                st.success(f"✅ Found {len(opportunities)} arbitrage opportunities!")

                for i, opp in enumerate(opportunities[:5]):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.markdown(f"**{i+1}. {opp['direction']}**")

                    with col2:
                        st.markdown(
                            f"Buy: ${opp['buy_price']:,.2f} | Sell: ${opp['sell_price']:,.2f}"
                        )

                    with col3:
                        st.markdown(
                            f"<span class='metric-positive'>+{opp['spread_pct']:.3f}%</span>",
                            unsafe_allow_html=True,
                        )

                # Arbitrage chart
                if len(opportunities) > 0:
                    opp_df = pd.DataFrame(opportunities[:10])

                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=opp_df["direction"],
                            y=opp_df["spread_pct"],
                            marker_color="green",
                            text=[f"{s:.3f}%" for s in opp_df["spread_pct"]],
                            textposition="auto",
                        )
                    )

                    fig.update_layout(
                        title="Top Arbitrage Opportunities",
                        xaxis_title="Trade Direction",
                        yaxis_title="Spread (%)",
                        height=400,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(
                    "No profitable arbitrage opportunities detected at current prices."
                )

        # TAB 5: Analytics
        with tab5:
            st.markdown("## Market Microstructure Analytics")

            if stats:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 24-Hour Statistics")
                    stats_df = pd.DataFrame(
                        [
                            {
                                "Metric": "Last Price",
                                "Value": f"${stats.get('last_price', 0):,.2f}",
                            },
                            {
                                "Metric": "24h High",
                                "Value": f"${stats.get('high', 0):,.2f}",
                            },
                            {
                                "Metric": "24h Low",
                                "Value": f"${stats.get('low', 0):,.2f}",
                            },
                            {
                                "Metric": "24h Volume",
                                "Value": f"{stats.get('volume', 0):,.2f}",
                            },
                            {
                                "Metric": "Quote Volume",
                                "Value": f"${stats.get('quote_volume', 0):,.0f}",
                            },
                        ]
                    )
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("### Price Range")

                    fig = go.Figure()

                    # Price range
                    high = stats.get("high", 0)
                    low = stats.get("low", 0)
                    current = stats.get("last_price", 0)

                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=current,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Current Price Position"},
                            gauge={
                                "axis": {"range": [low, high]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {
                                        "range": [low, (high + low) / 2],
                                        "color": "lightgray",
                                    },
                                    {
                                        "range": [(high + low) / 2, high],
                                        "color": "gray",
                                    },
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": current,
                                },
                            },
                        )
                    )

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(
            "Failed to fetch live data. Please check your internet connection and API limits."
        )

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
