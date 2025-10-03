"""
Kafka producer for streaming order book data.

Integrates with WebSocket clients to publish real-time order book snapshots
to Kafka topics for downstream processing.
"""

import json
import time
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger

from ingestion.websocket_client import (
    OrderBookSnapshot,
    BinanceOrderBookClient,
    CoinbaseOrderBookClient,
)


class OrderBookKafkaProducer:
    """
    Kafka producer for order book data streaming.

    Publishes order book snapshots to Kafka topics with:
    - Automatic serialization
    - Error handling and retries
    - Performance metrics tracking
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "order-book-snapshots",
        compression_type: str = "snappy",
        acks: str = "all",
        retries: int = 3,
        **kwargs,
    ):
        """
        Initialize Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic to publish to
            compression_type: Compression algorithm (snappy, gzip, lz4)
            acks: Acknowledgment level (0, 1, 'all')
            retries: Number of retries on failure
        """
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers

        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            compression_type=compression_type,
            acks=acks,
            retries=retries,
            linger_ms=kwargs.get("linger_ms", 10),
            batch_size=kwargs.get("batch_size", 16384),
            max_in_flight_requests_per_connection=kwargs.get("max_in_flight", 5),
        )

        # Metrics
        self.messages_sent = 0
        self.errors = 0
        self.start_time = time.time()

        logger.info(f"Kafka producer initialized: {bootstrap_servers} -> {topic}")

    def on_success(self, record_metadata):
        """Callback for successful message delivery."""
        self.messages_sent += 1
        if self.messages_sent % 1000 == 0:
            elapsed = time.time() - self.start_time
            rate = self.messages_sent / elapsed
            logger.info(f"Sent {self.messages_sent:,} messages ({rate:.2f} msg/sec)")

    def on_error(self, exception):
        """Callback for message delivery errors."""
        self.errors += 1
        logger.error(f"Error sending message: {exception}")

    def send_snapshot(self, snapshot: OrderBookSnapshot) -> bool:
        """
        Send order book snapshot to Kafka.

        Args:
            snapshot: OrderBookSnapshot instance

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message key from exchange and symbol
            key = f"{snapshot.exchange}:{snapshot.symbol}"

            # Convert snapshot to dict
            message = snapshot.to_dict()

            # Send to Kafka
            future = self.producer.send(
                self.topic, key=key, value=message, timestamp_ms=snapshot.timestamp
            )

            # Add callbacks
            future.add_callback(self.on_success)
            future.add_errback(self.on_error)

            return True

        except Exception as e:
            logger.error(f"Error sending snapshot to Kafka: {e}")
            self.errors += 1
            return False

    def flush(self):
        """Flush pending messages."""
        self.producer.flush()
        logger.info(f"Flushed {self.messages_sent:,} messages")

    def close(self):
        """Close the Kafka producer."""
        self.flush()
        self.producer.close()
        logger.info(
            f"Kafka producer closed. Total sent: {self.messages_sent:,}, Errors: {self.errors}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        elapsed = time.time() - self.start_time
        return {
            "messages_sent": self.messages_sent,
            "errors": self.errors,
            "elapsed_seconds": elapsed,
            "messages_per_second": self.messages_sent / elapsed if elapsed > 0 else 0,
            "error_rate": (
                self.errors / self.messages_sent if self.messages_sent > 0 else 0
            ),
        }


class StreamingPipeline:
    """
    Complete streaming pipeline: WebSocket → Kafka.

    Connects to exchange WebSocket, receives order book data,
    and streams to Kafka topics.
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        kafka_bootstrap_servers: str = "localhost:9092",
        kafka_topic: str = "order-book-snapshots",
        **kwargs,
    ):
        """
        Initialize streaming pipeline.

        Args:
            exchange: Exchange name ('binance' or 'coinbase')
            symbol: Trading symbol
            kafka_bootstrap_servers: Kafka broker addresses
            kafka_topic: Kafka topic name
        """
        self.exchange = exchange
        self.symbol = symbol

        # Initialize Kafka producer
        self.kafka_producer = OrderBookKafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers, topic=kafka_topic, **kwargs
        )

        # Initialize WebSocket client
        if exchange.lower() == "binance":
            self.ws_client = BinanceOrderBookClient(
                symbol=symbol,
                on_message_callback=self.on_orderbook_update,
                depth_levels=kwargs.get("depth_levels", 20),
                update_speed=kwargs.get("update_speed", "100ms"),
            )
        elif exchange.lower() == "coinbase":
            self.ws_client = CoinbaseOrderBookClient(
                symbol=symbol, on_message_callback=self.on_orderbook_update
            )
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

        logger.info(
            f"Streaming pipeline initialized: {exchange}:{symbol} -> Kafka:{kafka_topic}"
        )

    def on_orderbook_update(self, snapshot: OrderBookSnapshot):
        """Callback for order book updates - send to Kafka."""
        self.kafka_producer.send_snapshot(snapshot)

    def start(self):
        """Start the streaming pipeline."""
        logger.info(f"Starting streaming pipeline for {self.exchange}:{self.symbol}")
        try:
            self.ws_client.start()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stop()

    def stop(self):
        """Stop the streaming pipeline."""
        logger.info("Stopping streaming pipeline")
        self.ws_client.stop()
        self.kafka_producer.close()

        # Print final stats
        stats = self.kafka_producer.get_stats()
        logger.info(f"Pipeline Statistics: {stats}")


# Multi-exchange streaming manager
class MultiExchangeStreamer:
    """
    Manage multiple exchange streams simultaneously.

    Streams from multiple exchanges/symbols to Kafka topics.
    """

    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092"):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.pipelines = []

    def add_stream(
        self, exchange: str, symbol: str, topic: Optional[str] = None, **kwargs
    ):
        """Add a new streaming pipeline."""
        topic = topic or "order-book-snapshots"

        pipeline = StreamingPipeline(
            exchange=exchange,
            symbol=symbol,
            kafka_bootstrap_servers=self.kafka_bootstrap_servers,
            kafka_topic=topic,
            **kwargs,
        )

        self.pipelines.append(pipeline)
        logger.info(f"Added stream: {exchange}:{symbol} -> {topic}")

    def start_all(self):
        """Start all streaming pipelines (non-blocking)."""
        import threading

        threads = []
        for pipeline in self.pipelines:
            thread = threading.Thread(target=pipeline.start, daemon=True)
            thread.start()
            threads.append(thread)

        logger.info(f"Started {len(threads)} streaming pipelines")

        # Wait for threads
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            self.stop_all()

    def stop_all(self):
        """Stop all streaming pipelines."""
        for pipeline in self.pipelines:
            pipeline.stop()


# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Order Book Kafka Streaming Pipeline")
    parser.add_argument(
        "--exchange",
        choices=["binance", "coinbase"],
        required=True,
        help="Exchange name",
    )
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument(
        "--kafka-servers", default="localhost:9092", help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--kafka-topic", default="order-book-snapshots", help="Kafka topic"
    )
    parser.add_argument(
        "--depth", type=int, default=20, help="Depth levels (Binance only)"
    )
    parser.add_argument(
        "--update-speed", default="100ms", help="Update speed (Binance only)"
    )

    args = parser.parse_args()

    # Create and start pipeline
    pipeline = StreamingPipeline(
        exchange=args.exchange,
        symbol=args.symbol,
        kafka_bootstrap_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        depth_levels=args.depth,
        update_speed=args.update_speed,
    )

    try:
        pipeline.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        pipeline.stop()
