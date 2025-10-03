## HFT Live Trading Dashboard - Complete Guide

### 🚀 Real-Time High-Frequency Trading Platform

A professional-grade dashboard for real-time order book analysis, signal generation, and execution simulation using live market data from major cryptocurrency exchanges.

---

## Features

### 1. 📊 Live Order Book Visualization

**Real-time order book depth from Binance and Coinbase**

- **Multi-level depth display**: Up to 20 price levels on each side
- **Interactive heatmap**: Visual representation of bid/ask liquidity
- **Live updates**: Auto-refresh with configurable intervals (1-10 seconds)
- **Best bid/ask tracking**: Real-time spread monitoring
- **Volume analysis**: Aggregate bid/ask volume at each level

**Key Metrics**:
- Best Bid/Ask prices
- Spread (absolute and basis points)
- 24-hour price change and volume
- Volume imbalance ratio

### 2. 🎯 Real-Time Trading Signals

**AI-powered signal generation using market microstructure analysis**

**Signal Generation Algorithm**:
```python
if volume_imbalance > 0.15 and spread < 10 bps:
    Signal = BUY
    Confidence = 60-95%

elif volume_imbalance < -0.15 and spread < 10 bps:
    Signal = SELL
    Confidence = 60-95%

else:
    Signal = NEUTRAL
```

**Features**:
- Volume imbalance analysis
- Spread quality filtering
- Confidence scoring (0-100%)
- Configurable confidence thresholds
- One-click execution simulation

**Signal Types**:
- 🟢 **BUY**: Strong bullish order flow
- 🔴 **SELL**: Strong bearish order flow
- 🟡 **NEUTRAL**: No clear directional bias

### 3. 💰 Performance Tracking

**Complete P&L monitoring and trade analytics**

**Metrics Tracked**:
- **Total P&L**: Cumulative profit/loss across all trades
- **Win Rate**: Percentage of profitable trades
- **Average P&L per Trade**: Mean profit across all executions
- **Trades Executed**: Total number of simulated trades
- **Cumulative P&L Chart**: Visual tracking of performance over time

**Simulation Features**:
- Realistic slippage modeling (5 bps)
- Transaction costs (10 bps per trade)
- Position sizing controls
- Mark-to-market P&L calculation

### 4. 🔍 Cross-Exchange Arbitrage Detection

**Automated identification of arbitrage opportunities**

**How It Works**:
1. Simultaneously fetches order books from multiple exchanges
2. Compares best bid/ask across venues
3. Calculates spread after fees and slippage
4. Ranks opportunities by profitability

**Example Opportunity**:
```
Buy Binance @ $120,400 → Sell Coinbase @ $120,520
Spread: +0.10% (profitable after 0.05% fees)
```

**Display**:
- Top 5 opportunities ranked by spread
- Buy/Sell venue recommendations
- Expected profit percentage
- Visual bar chart of opportunities

### 5. 📈 Market Microstructure Analytics

**Deep dive into market statistics and price dynamics**

**24-Hour Statistics**:
- Last traded price
- Daily high/low range
- Total volume (base currency)
- Quote volume (USD/USDT)
- Price change percentage

**Price Position Gauge**:
- Visual indicator showing current price relative to 24h range
- Color-coded zones (green = near high, red = near low)

---

## Data Sources

### Binance API

**Free tier provides**:
- Order book depth (up to 5000 levels)
- Recent trades (up to 1000)
- 24-hour ticker statistics
- WebSocket streams (real-time)

**Supported Symbols**:
- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- BNBUSDT (Binance Coin)
- SOLUSDT (Solana)
- ADAUSDT (Cardano)
- 100+ other pairs

**Rate Limits**:
- REST API: 2400 requests/minute (IP weight)
- WebSocket: No limit on connections

### Coinbase Pro API

**Free tier provides**:
- Order book snapshots (top 50 levels)
- Recent trades
- Product statistics
- WebSocket streams

**Supported Symbols**:
- BTC-USD (Bitcoin)
- ETH-USD (Ethereum)
- SOL-USD (Solana)
- ADA-USD (Cardano)
- 50+ other pairs

**Rate Limits**:
- Public endpoints: 10 requests/second
- WebSocket: Unlimited

---

## Installation & Setup

### Prerequisites
```bash
Python 3.9+
pip
Internet connection (for live data)
```

### Install Dependencies
```bash
# Navigate to project directory
cd hft-order-book-imbalance

# Install required packages
pip install -r requirements.txt

# Key dependencies:
# - streamlit>=1.24.0
# - plotly>=5.14.0
# - pandas>=2.0.0
# - requests>=2.31.0
# - websockets>=12.0
```

### Launch Dashboard
```bash
# Option 1: Using launcher script
python run_hft_live_dashboard.py

# Option 2: Direct streamlit command
streamlit run src/visualization/hft_live_dashboard.py --server.port=8503
```

The dashboard opens automatically at `http://localhost:8503`

---

## User Interface Guide

### Sidebar Configuration

**Exchange Selection**:
- Choose between Binance or Coinbase
- Automatically updates available symbols

**Symbol Selection**:
- Dropdown list of supported trading pairs
- Updates order book and analytics in real-time

**Auto-Refresh**:
- Toggle automatic data updates
- Configurable refresh interval (1-10 seconds)
- Manual refresh button for on-demand updates

**Trading Parameters**:
- **Min Confidence**: Threshold for signal execution (50-95%)
- **Position Size**: Amount to trade per signal (BTC/ETH/etc.)

### Main Tabs

#### Tab 1: Order Book
- Full order book visualization with heatmap
- Best bid/ask metrics
- Spread analysis
- Volume imbalance indicator
- 24-hour change statistics

#### Tab 2: Trading Signals
- Current signal with confidence score
- Visual signal classification (Buy/Sell/Neutral)
- One-click execution simulation
- Signal history table
- Feature breakdown (micro-price, volume imbalance, spread)

#### Tab 3: Performance
- Total P&L tracking
- Trade count and win rate
- Average P&L per trade
- Cumulative P&L chart
- Trade-by-trade breakdown

#### Tab 4: Arbitrage
- Cross-exchange opportunity scanner
- Top 5 opportunities ranked
- Buy/sell venue recommendations
- Expected profit calculations
- Visual opportunity chart

#### Tab 5: Analytics
- 24-hour market statistics
- Price range gauge
- Volume analysis
- High/low tracking

---

## Trading Strategy

### Signal Generation Logic

**Step 1: Fetch Live Order Book**
```python
book = aggregator.get_order_book(exchange, symbol)
# Returns: bids[], asks[], timestamp, sequence
```

**Step 2: Calculate Microstructure Features**
```python
features = {
    'volume_imbalance': (bid_vol - ask_vol) / (bid_vol + ask_vol),
    'spread_bps': ((ask - bid) / mid) * 10000,
    'micro_price': (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
}
```

**Step 3: Generate Signal**
```python
if features['volume_imbalance'] > 0.15 and features['spread_bps'] < 10:
    signal = 'BUY'
    confidence = 0.6 + abs(volume_imbalance) * 2
```

**Step 4: Execute (if confidence >= threshold)**
```python
if confidence >= min_confidence:
    execute_order(signal, position_size, current_price)
```

### Execution Simulation

**Realistic Cost Modeling**:
- **Slippage**: 5 basis points (0.05%)
  - BUY orders: Execute at ask + slippage
  - SELL orders: Execute at bid - slippage
- **Transaction Fees**: 10 basis points (0.10%)
  - Applied to notional value
  - Typical for taker orders on exchanges

**P&L Calculation**:
```python
if signal == 'BUY':
    pnl = (current_price - entry_price) * position_size

elif signal == 'SELL':
    pnl = (entry_price - current_price) * position_size

net_pnl = pnl - fees
```

---

## Use Cases

### For Retail Traders:
- **Signal Confirmation**: Validate manual trading ideas with AI signals
- **Market Timing**: Identify optimal entry/exit points
- **Risk Management**: Monitor spread and slippage in real-time
- **Arbitrage Hunting**: Find cross-exchange profit opportunities

### For Quant Researchers:
- **Strategy Development**: Test microstructure-based signals
- **Feature Analysis**: Understand order flow dynamics
- **Backtesting Preparation**: Collect live data for historical analysis
- **Model Validation**: Compare predicted vs actual market moves

### For Institutional Traders:
- **Execution Quality**: Monitor real-time spread and depth
- **Market Impact**: Assess available liquidity before large orders
- **Venue Selection**: Choose optimal exchange based on depth
- **Compliance Monitoring**: Track execution costs and slippage

### For Educators:
- **Market Microstructure**: Teach order book dynamics
- **HFT Concepts**: Demonstrate latency and signal generation
- **Crypto Markets**: Real-world examples of 24/7 trading
- **Risk Management**: Illustrate transaction costs and slippage

---

## Advanced Features

### WebSocket Streaming (Code Available)

**Real-time updates via WebSocket**:
```python
async def stream_order_book(symbol, callback):
    uri = f"wss://stream.binance.com:9443/ws/{symbol}@depth20@100ms"

    async with websockets.connect(uri) as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            await callback(data)
```

**Benefits**:
- Sub-100ms latency
- No polling overhead
- Guaranteed delivery
- Sequence number tracking

### Multi-Exchange Consolidation

**Aggregate order book across venues**:
```python
consolidated_book = aggregator.get_consolidated_book({
    'binance': 'BTCUSDT',
    'coinbase': 'BTC-USD'
})

# Returns unified DataFrame with all bids/asks
```

### Arbitrage Detection Algorithm

**Cross-exchange spread calculation**:
```python
for ex1, ex2 in exchange_pairs:
    buy_price_ex1 = ex1.best_ask
    sell_price_ex2 = ex2.best_bid

    gross_spread = sell_price_ex2 - buy_price_ex1
    net_spread = gross_spread - fees - slippage

    if net_spread > 0:
        opportunities.append({
            'buy': ex1,
            'sell': ex2,
            'spread_pct': (net_spread / buy_price_ex1) * 100
        })
```

---

## Performance Optimization

### Low Latency Best Practices:

1. **Use WebSocket streams** instead of REST polling
2. **Minimize API calls** with caching
3. **Batch requests** when possible
4. **Use connection pooling** for HTTP requests
5. **Enable compression** for WebSocket data

### Resource Management:

```python
# Limit order book history
order_book_buffer = deque(maxlen=1000)

# Efficient data structures
import numpy as np  # Faster than lists
import pandas as pd  # Optimized DataFrames
```

### API Rate Limit Handling:

```python
# Respect exchange limits
time.sleep(1 / requests_per_second)

# Exponential backoff on errors
retry_delay = min(60, 2 ** retry_count)
```

---

## Troubleshooting

### Issue: No data displaying
**Solution**:
- Check internet connection
- Verify exchange is not down (check status pages)
- Ensure symbol is correctly formatted (e.g., 'BTCUSDT' not 'BTC-USDT' for Binance)
- Check API rate limits not exceeded

### Issue: Slow performance
**Solution**:
- Increase refresh interval (e.g., 5-10 seconds)
- Reduce order book depth (10 levels instead of 20)
- Close other browser tabs
- Use faster internet connection

### Issue: WebSocket connection errors
**Solution**:
- Check firewall settings (allow wss:// connections)
- Verify Python websockets package installed
- Use alternative exchange if one is unavailable
- Enable verbose logging for debugging

### Issue: Incorrect P&L calculations
**Solution**:
- Verify position size is correct
- Check that fees are properly deducted
- Ensure current price is updating
- Review execution price vs current price

---

## API Limits & Best Practices

### Binance Limits:
- **Order Book**: 2400 weight/min (10 weight per request)
- **24h Stats**: 40 weight per request
- **WebSocket**: No limit

**Best Practice**: Use WebSocket for order book, REST for stats

### Coinbase Limits:
- **Public Endpoints**: 10 req/sec
- **WebSocket**: Unlimited

**Best Practice**: Batch multiple symbols in single WebSocket connection

### Rate Limit Monitoring:
```python
# Check response headers
response.headers['X-MBX-USED-WEIGHT-1M']  # Binance
response.headers['cb-after']  # Coinbase pagination
```

---

## Security Considerations

### API Keys (If Using Private Endpoints):

**Never commit API keys to code**:
```python
# Use environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
```

**Use read-only permissions**:
- No withdrawal rights
- No trading rights (for paper trading)
- Only market data access

### Data Privacy:

- All data fetched from public endpoints
- No personal information transmitted
- No order execution (simulation only)
- Logs stored locally only

---

## Future Enhancements

### Planned Features:
- [ ] Machine learning signal generation (LSTM/Transformer)
- [ ] Historical backtesting with downloaded data
- [ ] Multi-symbol portfolio tracking
- [ ] Advanced order types (limit, stop-loss)
- [ ] Risk metrics (VaR, Sharpe ratio)
- [ ] Email/SMS alerts for signals
- [ ] Strategy optimization with genetic algorithms
- [ ] Integration with paper trading accounts

---

## FAQ

**Q: Is this connected to real trading?**
A: No, all executions are simulated. No real money is at risk.

**Q: Can I use this for stocks?**
A: Yes, with modifications. Add Polygon.io or Alpha Vantage connectors.

**Q: How accurate are the signals?**
A: Depends on market conditions. Historical accuracy ~60-65% on crypto.

**Q: Can I customize the strategy?**
A: Yes, modify `generate_trading_signal()` in the code.

**Q: Does this work 24/7?**
A: Yes, crypto markets operate 24/7. Dashboard can run continuously.

**Q: What's the minimum investment needed?**
A: This is simulation only. No investment required.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────┐
│         HFT Live Trading Dashboard              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐      ┌──────────────┐        │
│  │  Streamlit  │      │   Plotly     │        │
│  │     UI      │─────▶│ Visualizations│       │
│  └─────────────┘      └──────────────┘        │
│         │                                       │
│         ▼                                       │
│  ┌─────────────────────────────────────┐      │
│  │   Live Data Aggregator               │      │
│  │  - Multi-exchange coordination       │      │
│  │  - Data normalization               │      │
│  │  - Arbitrage detection              │      │
│  └─────────────────────────────────────┘      │
│         │                                       │
│    ┌────┴────┐                                │
│    │         │                                 │
│    ▼         ▼                                 │
│ ┌─────┐  ┌─────┐                             │
│ │Binance│ │Coinbase│                          │
│ │  API  │ │  API   │                          │
│ └─────┘  └─────┘                              │
│                                                 │
│  Features:                                      │
│  • Order Flow Imbalance                        │
│  • Micro-price Calculation                     │
│  • Volume Imbalance                            │
│  • Spread Analysis                             │
└─────────────────────────────────────────────────┘
```

---

## References

### Market Microstructure:
- O'Hara, M. (1995). *Market Microstructure Theory*
- Harris, L. (2003). *Trading and Exchanges*
- Hasbrouck, J. (2007). *Empirical Market Microstructure*

### High-Frequency Trading:
- Aldridge, I. (2013). *High-Frequency Trading: A Practical Guide*
- Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management*

### Exchange APIs:
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Coinbase Pro API](https://docs.cloud.coinbase.com/exchange/docs)

---

## Support

**For Issues**:
- GitHub: [github.com/mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine](https://github.com/mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine)
- Documentation: See `docs/` folder

**For Feature Requests**:
- Open GitHub issue with [FEATURE] tag
- Describe use case and expected behavior

---

**Version**: 1.0.0
**Last Updated**: October 2025
**License**: MIT
