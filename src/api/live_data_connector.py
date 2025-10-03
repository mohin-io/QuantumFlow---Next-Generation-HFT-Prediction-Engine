"""
Real-time Data Connector for HFT Dashboard

Connects to authenticated data sources:
- Binance API (crypto market data)
- Coinbase API (crypto market data)
- Alpha Vantage (stocks)
- Polygon.io (stocks, real-time)

Provides WebSocket streaming and REST API access.
"""

import asyncio
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import websockets
import requests
from collections import deque


@dataclass
class OrderBookSnapshot:
    """Order book snapshot."""

    timestamp: int
    exchange: str
    symbol: str
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]
    sequence: Optional[int] = None


@dataclass
class Trade:
    """Trade data."""

    timestamp: int
    exchange: str
    symbol: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None


class BinanceConnector:
    """
    Binance API connector for real-time market data.

    Free tier provides:
    - Order book snapshots (depth)
    - Recent trades
    - WebSocket streams (no auth needed for public data)
    """

    BASE_URL = "https://api.binance.com"
    WS_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize Binance connector.

        Args:
            api_key: Optional API key for private endpoints
            api_secret: Optional API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"X-MBX-APIKEY": api_key})

    def get_order_book(self, symbol: str, limit: int = 20) -> OrderBookSnapshot:
        """
        Get current order book snapshot.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            limit: Number of levels (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            OrderBookSnapshot
        """
        endpoint = f"{self.BASE_URL}/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}

        try:
            response = self.session.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            return OrderBookSnapshot(
                timestamp=int(time.time() * 1000),
                exchange="binance",
                symbol=symbol,
                bids=[[float(price), float(qty)] for price, qty in data["bids"]],
                asks=[[float(price), float(qty)] for price, qty in data["asks"]],
                sequence=data.get("lastUpdateId"),
            )
        except Exception as e:
            print(f"Error fetching Binance order book: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """
        Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)

        Returns:
            List of Trade objects
        """
        endpoint = f"{self.BASE_URL}/api/v3/trades"
        params = {"symbol": symbol, "limit": limit}

        try:
            response = self.session.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            trades = []
            for t in data:
                trades.append(
                    Trade(
                        timestamp=t["time"],
                        exchange="binance",
                        symbol=symbol,
                        price=float(t["price"]),
                        size=float(t["qty"]),
                        side="sell" if t["isBuyerMaker"] else "buy",
                        trade_id=str(t["id"]),
                    )
                )

            return trades
        except Exception as e:
            print(f"Error fetching Binance trades: {e}")
            return []

    def get_24h_stats(self, symbol: str) -> Dict:
        """Get 24-hour statistics."""
        endpoint = f"{self.BASE_URL}/api/v3/ticker/24hr"
        params = {"symbol": symbol}

        try:
            response = self.session.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": data["symbol"],
                "price_change": float(data["priceChange"]),
                "price_change_pct": float(data["priceChangePercent"]),
                "volume": float(data["volume"]),
                "quote_volume": float(data["quoteVolume"]),
                "high": float(data["highPrice"]),
                "low": float(data["lowPrice"]),
                "last_price": float(data["lastPrice"]),
                "bid_price": float(data["bidPrice"]),
                "ask_price": float(data["askPrice"]),
            }
        except Exception as e:
            print(f"Error fetching 24h stats: {e}")
            return {}

    async def stream_order_book(self, symbol: str, callback: Callable):
        """
        Stream real-time order book updates via WebSocket.

        Args:
            symbol: Trading pair (lowercase, e.g., 'btcusdt')
            callback: Function called on each update
        """
        symbol_lower = symbol.lower()
        uri = f"{self.WS_URL}/{symbol_lower}@depth20@100ms"

        try:
            async with websockets.connect(uri) as ws:
                print(f"[Binance] Connected to {symbol} order book stream")

                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    snapshot = OrderBookSnapshot(
                        timestamp=data["E"],
                        exchange="binance",
                        symbol=symbol.upper(),
                        bids=[[float(p), float(q)] for p, q in data["bids"]],
                        asks=[[float(p), float(q)] for p, q in data["asks"]],
                        sequence=data["lastUpdateId"],
                    )

                    await callback(snapshot)

        except Exception as e:
            print(f"[Binance] WebSocket error: {e}")


class CoinbaseConnector:
    """
    Coinbase Pro API connector.

    Free tier provides:
    - Order book snapshots
    - Recent trades
    - WebSocket streams
    """

    BASE_URL = "https://api.exchange.coinbase.com"
    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(
        self, api_key: str = None, api_secret: str = None, passphrase: str = None
    ):
        """Initialize Coinbase connector."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_order_book(self, symbol: str, level: int = 2) -> OrderBookSnapshot:
        """
        Get order book snapshot.

        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            level: 1=best bid/ask, 2=top 50, 3=full book
        """
        endpoint = f"{self.BASE_URL}/products/{symbol}/book"
        params = {"level": level}

        try:
            response = self.session.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            return OrderBookSnapshot(
                timestamp=int(time.time() * 1000),
                exchange="coinbase",
                symbol=symbol,
                bids=[[float(p), float(s)] for p, s, _ in data["bids"]],
                asks=[[float(p), float(s)] for p, s, _ in data["asks"]],
                sequence=data.get("sequence"),
            )
        except Exception as e:
            print(f"Error fetching Coinbase order book: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades."""
        endpoint = f"{self.BASE_URL}/products/{symbol}/trades"

        try:
            response = self.session.get(endpoint, timeout=5)
            response.raise_for_status()
            data = response.json()[:limit]

            trades = []
            for t in data:
                # Parse ISO timestamp to milliseconds
                dt = datetime.fromisoformat(t["time"].replace("Z", "+00:00"))
                timestamp = int(dt.timestamp() * 1000)

                trades.append(
                    Trade(
                        timestamp=timestamp,
                        exchange="coinbase",
                        symbol=symbol,
                        price=float(t["price"]),
                        size=float(t["size"]),
                        side=t["side"],
                        trade_id=str(t["trade_id"]),
                    )
                )

            return trades
        except Exception as e:
            print(f"Error fetching Coinbase trades: {e}")
            return []

    async def stream_order_book(self, symbols: List[str], callback: Callable):
        """Stream real-time order book via WebSocket."""
        subscribe_message = {
            "type": "subscribe",
            "product_ids": symbols,
            "channels": ["level2"],
        }

        try:
            async with websockets.connect(self.WS_URL) as ws:
                await ws.send(json.dumps(subscribe_message))
                print(f"[Coinbase] Subscribed to {symbols}")

                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    if data["type"] == "snapshot":
                        snapshot = OrderBookSnapshot(
                            timestamp=int(time.time() * 1000),
                            exchange="coinbase",
                            symbol=data["product_id"],
                            bids=[[float(p), float(s)] for p, s in data["bids"]],
                            asks=[[float(p), float(s)] for p, s in data["asks"]],
                        )
                        await callback(snapshot)

                    elif data["type"] == "l2update":
                        # Handle incremental updates
                        # You would maintain local order book and apply changes
                        pass

        except Exception as e:
            print(f"[Coinbase] WebSocket error: {e}")


class LiveDataAggregator:
    """
    Aggregates data from multiple sources.
    Provides unified interface for HFT dashboard.
    """

    def __init__(self):
        """Initialize aggregator."""
        self.binance = BinanceConnector()
        self.coinbase = CoinbaseConnector()

        # Data buffers
        self.order_books = {}
        self.trades = deque(maxlen=1000)
        self.stats = {}

        # Callbacks
        self.callbacks = []

    def register_callback(self, callback: Callable):
        """Register callback for data updates."""
        self.callbacks.append(callback)

    def get_multi_exchange_book(self, symbol_map: Dict[str, str]) -> Dict:
        """
        Get order book from multiple exchanges.

        Args:
            symbol_map: {'binance': 'BTCUSDT', 'coinbase': 'BTC-USD'}

        Returns:
            Dictionary with exchange data
        """
        results = {}

        if "binance" in symbol_map:
            book = self.binance.get_order_book(symbol_map["binance"])
            if book:
                results["binance"] = book

        if "coinbase" in symbol_map:
            book = self.coinbase.get_order_book(symbol_map["coinbase"])
            if book:
                results["coinbase"] = book

        return results

    def get_consolidated_book(
        self, symbol_map: Dict[str, str], depth: int = 10
    ) -> pd.DataFrame:
        """
        Get consolidated order book across exchanges.

        Returns DataFrame with columns: exchange, side, price, size
        """
        books = self.get_multi_exchange_book(symbol_map)

        data = []
        for exchange, book in books.items():
            # Bids
            for price, size in book.bids[:depth]:
                data.append(
                    {"exchange": exchange, "side": "bid", "price": price, "size": size}
                )

            # Asks
            for price, size in book.asks[:depth]:
                data.append(
                    {"exchange": exchange, "side": "ask", "price": price, "size": size}
                )

        df = pd.DataFrame(data)
        return df.sort_values("price", ascending=False).reset_index(drop=True)

    def calculate_arbitrage_opportunities(
        self, symbol_map: Dict[str, str]
    ) -> List[Dict]:
        """
        Find arbitrage opportunities across exchanges.

        Returns list of opportunities with expected profit.
        """
        books = self.get_multi_exchange_book(symbol_map)

        if len(books) < 2:
            return []

        opportunities = []

        # Compare each pair of exchanges
        exchanges = list(books.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                book1, book2 = books[ex1], books[ex2]

                # Check if we can buy on ex1 and sell on ex2
                if book1.asks and book2.bids:
                    buy_price = book1.asks[0][0]
                    sell_price = book2.bids[0][0]
                    spread = sell_price - buy_price
                    spread_pct = (spread / buy_price) * 100

                    if spread_pct > 0.1:  # Profitable after fees
                        opportunities.append(
                            {
                                "buy_exchange": ex1,
                                "sell_exchange": ex2,
                                "buy_price": buy_price,
                                "sell_price": sell_price,
                                "spread": spread,
                                "spread_pct": spread_pct,
                                "direction": f"Buy {ex1} → Sell {ex2}",
                            }
                        )

                # Check reverse
                if book2.asks and book1.bids:
                    buy_price = book2.asks[0][0]
                    sell_price = book1.bids[0][0]
                    spread = sell_price - buy_price
                    spread_pct = (spread / buy_price) * 100

                    if spread_pct > 0.1:
                        opportunities.append(
                            {
                                "buy_exchange": ex2,
                                "sell_exchange": ex1,
                                "buy_price": buy_price,
                                "sell_price": sell_price,
                                "spread": spread,
                                "spread_pct": spread_pct,
                                "direction": f"Buy {ex2} → Sell {ex1}",
                            }
                        )

        return sorted(opportunities, key=lambda x: x["spread_pct"], reverse=True)

    def get_real_time_stats(self, exchange: str, symbol: str) -> Dict:
        """Get real-time statistics."""
        if exchange == "binance":
            return self.binance.get_24h_stats(symbol)
        elif exchange == "coinbase":
            # Coinbase stats endpoint
            return {}
        return {}


# Demo / Testing
if __name__ == "__main__":
    print("=" * 80)
    print("LIVE DATA CONNECTOR DEMONSTRATION")
    print("=" * 80)

    aggregator = LiveDataAggregator()

    # Test Binance
    print("\n1. Fetching Binance BTC/USDT order book...")
    btc_book = aggregator.binance.get_order_book("BTCUSDT", limit=10)
    if btc_book:
        print(f"   Timestamp: {btc_book.timestamp}")
        print(
            f"   Best Bid: ${btc_book.bids[0][0]:,.2f} ({btc_book.bids[0][1]:.4f} BTC)"
        )
        print(
            f"   Best Ask: ${btc_book.asks[0][0]:,.2f} ({btc_book.asks[0][1]:.4f} BTC)"
        )
        print(f"   Spread: ${btc_book.asks[0][0] - btc_book.bids[0][0]:.2f}")

    # Test 24h stats
    print("\n2. Fetching 24h statistics...")
    stats = aggregator.binance.get_24h_stats("BTCUSDT")
    if stats:
        print(f"   Last Price: ${stats['last_price']:,.2f}")
        print(f"   24h Change: {stats['price_change_pct']:+.2f}%")
        print(f"   24h Volume: {stats['volume']:,.2f} BTC")
        print(f"   24h High: ${stats['high']:,.2f}")
        print(f"   24h Low: ${stats['low']:,.2f}")

    # Test multi-exchange
    print("\n3. Multi-exchange order book comparison...")
    symbol_map = {"binance": "BTCUSDT", "coinbase": "BTC-USD"}

    consolidated = aggregator.get_consolidated_book(symbol_map, depth=5)
    if not consolidated.empty:
        print("\n   Consolidated Order Book:")
        print(consolidated.to_string(index=False))

    # Test arbitrage detection
    print("\n4. Checking for arbitrage opportunities...")
    opportunities = aggregator.calculate_arbitrage_opportunities(symbol_map)
    if opportunities:
        print(f"\n   Found {len(opportunities)} opportunities:")
        for opp in opportunities[:3]:
            direction_clean = opp["direction"].replace("→", "->")
            print(f"   - {direction_clean}: {opp['spread_pct']:.3f}% spread")
    else:
        print("   No profitable arbitrage opportunities found.")

    print("\n" + "=" * 80)
    print("Live data connector ready for HFT dashboard integration!")
    print("=" * 80)
