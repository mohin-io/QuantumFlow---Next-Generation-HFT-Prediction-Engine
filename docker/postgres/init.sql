-- Initialize HFT Order Book Database with TimescaleDB

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE hft_orderbook TO postgres;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS predictions;

-- Set search path
SET search_path TO public, market_data, features, predictions;

-- The actual tables will be created by SQLAlchemy models
-- This script just sets up the extensions and schemas
