"""
WebSocket client for collecting real-time order book data from cryptocurrency exchanges.

Supports:
- Binance (depth@100ms stream)
- Coinbase (level2 channel)
"""

import json
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import websocket
from loguru import logger


@dataclass
class OrderBookSnapshot:
    """Standardized order book snapshot format."""

    timestamp: int  # Unix timestamp in milliseconds
    exchange: str
    symbol: str
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]
    sequence: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @property
    def mid_price(self) -> float:
        """Calculate mid-price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000


class BaseOrderBookClient(ABC):
    """Base class for order book WebSocket clients."""

    def __init__(
        self,
        symbol: str,
        on_message_callback: Optional[Callable[[OrderBookSnapshot], None]] = None,
        reconnect_delay: int = 5,
        max_reconnects: int = 10,
    ):
        self.symbol = symbol
        self.on_message_callback = on_message_callback
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.reconnect_count = 0
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False

    @abstractmethod
    def get_websocket_url(self) -> str:
        """Return the WebSocket URL for the exchange."""
        pass

    @abstractmethod
    def parse_message(self, message: str) -> Optional[OrderBookSnapshot]:
        """Parse exchange-specific message format."""
        pass

    @abstractmethod
    def get_exchange_name(self) -> str:
        """Return the exchange name."""
        pass

    def on_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info(f"Connected to {self.get_exchange_name()} - {self.symbol}")
        self.reconnect_count = 0

    def on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            snapshot = self.parse_message(message)
            if snapshot and self.on_message_callback:
                self.on_message_callback(snapshot)
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            logger.debug(f"Raw message: {message}")

    def on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.warning(f"Connection closed: {close_status_code} - {close_msg}")

        if self.running and self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            logger.info(
                f"Reconnecting in {self.reconnect_delay}s (attempt {self.reconnect_count}/{self.max_reconnects})"
            )
            time.sleep(self.reconnect_delay)
            self.connect()
        else:
            logger.error("Max reconnection attempts reached or client stopped")
            self.running = False

    def connect(self):
        """Establish WebSocket connection."""
        url = self.get_websocket_url()
        logger.info(f"Connecting to {url}")

        self.ws = websocket.WebSocketApp(
            url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        self.running = True
        self.ws.run_forever()

    def start(self):
        """Start the WebSocket client."""
        logger.info(f"Starting {self.get_exchange_name()} client for {self.symbol}")
        self.connect()

    def stop(self):
        """Stop the WebSocket client."""
        logger.info(f"Stopping {self.get_exchange_name()} client")
        self.running = False
        if self.ws:
            self.ws.close()


class BinanceOrderBookClient(BaseOrderBookClient):
    """Binance order book WebSocket client.

    Connects to depth@100ms stream for real-time order book updates.
    """

    def __init__(
        self, symbol: str, depth_levels: int = 20, update_speed: str = "100ms", **kwargs
    ):
        """
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            depth_levels: Number of price levels (5, 10, 20)
            update_speed: Update frequency ('100ms' or '1000ms')
        """
        super().__init__(symbol, **kwargs)
        self.depth_levels = depth_levels
        self.update_speed = update_speed

    def get_exchange_name(self) -> str:
        return "Binance"

    def get_websocket_url(self) -> str:
        symbol_lower = self.symbol.lower()
        return f"wss://stream.binance.com:9443/ws/{symbol_lower}@depth{self.depth_levels}@{self.update_speed}"

    def parse_message(self, message: str) -> Optional[OrderBookSnapshot]:
        """Parse Binance depth update message."""
        try:
            data = json.loads(message)

            # Binance depth format
            if "lastUpdateId" not in data:
                return None

            timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Parse bids and asks
            bids = [[float(price), float(qty)] for price, qty in data.get("bids", [])]
            asks = [[float(price), float(qty)] for price, qty in data.get("asks", [])]

            return OrderBookSnapshot(
                timestamp=timestamp,
                exchange="binance",
                symbol=self.symbol,
                bids=bids,
                asks=asks,
                sequence=data.get("lastUpdateId"),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing Binance message: {e}")
            return None


class CoinbaseOrderBookClient(BaseOrderBookClient):
    """Coinbase Pro order book WebSocket client.

    Connects to level2 channel for order book snapshots and updates.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
        """
        super().__init__(symbol, **kwargs)
        self.order_book: Dict[str, List[List[float]]] = {"bids": [], "asks": []}

    def get_exchange_name(self) -> str:
        return "Coinbase"

    def get_websocket_url(self) -> str:
        return "wss://ws-feed.exchange.coinbase.com"

    def on_open(self, ws):
        """Subscribe to level2 channel on connection."""
        super().on_open(ws)

        subscribe_message = {
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channels": ["level2"],
        }
        ws.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to {self.symbol} level2 channel")

    def parse_message(self, message: str) -> Optional[OrderBookSnapshot]:
        """Parse Coinbase level2 message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            # Handle snapshot
            if msg_type == "snapshot":
                timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

                bids = [
                    [float(price), float(size)] for price, size in data.get("bids", [])
                ]
                asks = [
                    [float(price), float(size)] for price, size in data.get("asks", [])
                ]

                # Sort bids descending, asks ascending
                bids.sort(key=lambda x: x[0], reverse=True)
                asks.sort(key=lambda x: x[0])

                return OrderBookSnapshot(
                    timestamp=timestamp,
                    exchange="coinbase",
                    symbol=self.symbol,
                    bids=bids[:50],  # Top 50 levels
                    asks=asks[:50],
                    sequence=None,
                )

            # Handle l2update
            elif msg_type == "l2update":
                # For updates, we'd need to maintain state
                # For simplicity, we'll skip updates and rely on snapshots
                return None

            return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing Coinbase message: {e}")
            return None


def print_order_book_summary(snapshot: OrderBookSnapshot):
    """Helper function to print order book summary."""
    print(f"\n{'='*60}")
    print(f"Exchange: {snapshot.exchange.upper()} | Symbol: {snapshot.symbol}")
    print(
        f"Timestamp: {datetime.fromtimestamp(snapshot.timestamp/1000, tz=timezone.utc)}"
    )
    print(f"Mid Price: ${snapshot.mid_price:,.2f}")
    print(f"Spread: ${snapshot.spread:.4f} ({snapshot.spread_bps:.2f} bps)")
    print(f"\nTop 5 Levels:")
    print(f"{'BIDS':<30} | {'ASKS':<30}")
    print(f"{'-'*30}-+-{'-'*30}")

    for i in range(min(5, len(snapshot.bids), len(snapshot.asks))):
        bid_price, bid_qty = snapshot.bids[i] if i < len(snapshot.bids) else (0, 0)
        ask_price, ask_qty = snapshot.asks[i] if i < len(snapshot.asks) else (0, 0)
        print(
            f"${bid_price:>10.2f} x {bid_qty:>10.4f} | ${ask_price:>10.2f} x {ask_qty:>10.4f}"
        )


# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket Order Book Data Collector")
    parser.add_argument(
        "--exchange",
        choices=["binance", "coinbase"],
        required=True,
        help="Exchange name",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol (e.g., BTCUSDT for Binance, BTC-USD for Coinbase)",
    )
    parser.add_argument(
        "--depth", type=int, default=20, help="Depth levels for Binance (5, 10, 20)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    # Select client
    if args.exchange == "binance":
        client = BinanceOrderBookClient(
            symbol=args.symbol,
            depth_levels=args.depth,
            on_message_callback=print_order_book_summary,
        )
    else:  # coinbase
        client = CoinbaseOrderBookClient(
            symbol=args.symbol, on_message_callback=print_order_book_summary
        )

    try:
        client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        client.stop()
