# 📈 High-Frequency Order Book Imbalance Forecasting

> Advanced machine learning system for predicting short-term order book imbalances using computational statistics and deep learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Project Overview

This project implements a production-grade high-frequency trading (HFT) system that forecasts order book imbalances to predict short-term price movements. The system processes real-time market microstructure data, extracts sophisticated features, and deploys ensemble ML models for market making and execution algorithms.

**Key Capabilities:**
- Real-time order book data ingestion from multiple sources (NASDAQ, Binance, Coinbase)
- Advanced microstructure feature engineering (OFI, micro-price, queue dynamics)
- Multiple ML models: LSTM, Transformers, Bayesian Online Learning
- Ensemble meta-learner with dynamic weighting
- Production-ready API with <50ms latency
- Comprehensive backtesting with transaction cost modeling

## 📊 Key Results

*Results will be added as the project progresses*

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/mohin-io/hft-order-book-imbalance.git
cd hft-order-book-imbalance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Quick Start with Docker

```bash
# Start all services
docker-compose up -d

# Access the dashboard
# Open browser to http://localhost:8501

# API endpoint
# http://localhost:8000/docs
```

## 🏗️ Architecture

The system follows a modern data engineering and ML architecture:

```
Data Sources → Kafka Streams → Feature Engineering → ML Models → API/Dashboard
                     ↓
              PostgreSQL/InfluxDB
```

**Core Components:**
- **Data Ingestion**: WebSocket clients for real-time order book data
- **Streaming**: Kafka for high-throughput data pipelines
- **Storage**: PostgreSQL + TimescaleDB for time-series optimization
- **Feature Engineering**: Microstructure signals (OFI, micro-price, liquidity metrics)
- **ML Models**: LSTM, Transformers, Bayesian learners, Ensemble
- **Backtesting**: Walk-forward validation with transaction cost modeling
- **Deployment**: FastAPI service, Streamlit dashboard, Docker orchestration

See [docs/PLAN.md](docs/PLAN.md) for detailed architecture diagrams.

## 📈 Features

### Market Microstructure Signals
- **Order Flow Imbalance (OFI)**: Measures supply/demand pressure across price levels
- **Micro-price**: Volume-weighted fair value calculation
- **Volume Profiles**: Liquidity concentration and depth metrics
- **Queue Dynamics**: Order arrival rates, cancellation ratios
- **Realized Volatility**: Short-term volatility estimates

### Machine Learning Models
1. **LSTM/GRU Networks**: Sequence modeling of order book states
2. **Transformer Architecture**: Multi-head attention for long-range dependencies
3. **Bayesian Online Learning**: Real-time adaptive models with uncertainty quantification
4. **Ensemble Meta-learner**: Combines predictions across time horizons

### Backtesting & Evaluation
- Walk-forward validation
- Transaction cost analysis (spread, slippage, market impact)
- Performance metrics: Sharpe ratio, max drawdown, win rate
- Economic PnL simulation

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+ |
| **Data Processing** | Pandas, Polars, NumPy, PyArrow |
| **Streaming** | Apache Kafka, WebSockets |
| **Databases** | PostgreSQL, TimescaleDB, InfluxDB |
| **ML/DL** | PyTorch, TensorFlow, Scikit-learn, XGBoost |
| **Bayesian** | PyMC, Arviz |
| **Visualization** | Streamlit, Plotly, Seaborn, SHAP |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Orchestration** | Apache Airflow, Prefect |
| **Containerization** | Docker, Docker Compose |
| **Cloud** | AWS (ECS, RDS, S3, SageMaker) |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest, pytest-cov |

## 📁 Project Structure

```
HFT/
├── src/                    # Source code
│   ├── ingestion/         # Data collection modules
│   ├── features/          # Feature engineering
│   ├── models/            # ML model implementations
│   ├── backtesting/       # Backtesting engine
│   ├── visualization/     # Dashboard and plotting
│   └── api/               # FastAPI service
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Test suite
├── configs/               # Configuration files
├── docker/                # Docker configurations
├── airflow/               # Airflow DAGs
├── data/                  # Data storage (gitignored)
├── docs/                  # Documentation
└── README.md
```

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Quality
```bash
# Format code
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Starting Individual Services

```bash
# Data ingestion
python src/ingestion/websocket_client.py --exchange binance --symbol BTCUSDT

# Feature computation
python src/features/compute_features.py

# Model training
python src/models/train_lstm.py --config configs/lstm_config.yaml

# API server
uvicorn src.api.prediction_service:app --reload

# Dashboard
streamlit run src/visualization/dashboard.py
```

## 📚 Documentation

- [Implementation Plan](docs/PLAN.md) - Detailed development roadmap
- [Architecture Guide](docs/architecture/) - System design and diagrams
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
- [Research Notebooks](notebooks/) - Exploratory analysis and experiments

## 🎯 Performance Targets

- **Prediction Accuracy**: >55% (3-class classification)
- **Sharpe Ratio**: >1.5 on out-of-sample data
- **API Latency**: <50ms per prediction
- **Data Throughput**: >1000 ticks/second
- **Test Coverage**: >80%

## 🚧 Roadmap

- [x] Project setup and architecture design
- [ ] Data ingestion pipeline (Binance, Coinbase, LOBSTER)
- [ ] Feature engineering implementation
- [ ] Model development (LSTM, Transformer, Bayesian)
- [ ] Backtesting engine
- [ ] Visualization dashboard
- [ ] API deployment
- [ ] Cloud infrastructure (AWS)
- [ ] Performance optimization
- [ ] Documentation and testing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mohin Hasin**
- GitHub: [@mohin-io](https://github.com/mohin-io)
- Email: mohinhasin999@gmail.com

## 🙏 Acknowledgments

- Market microstructure research from leading quant finance papers
- Open-source ML frameworks (PyTorch, TensorFlow, Scikit-learn)
- Data sources: NASDAQ, Binance, Coinbase

---

⭐ Star this repository if you find it useful!
