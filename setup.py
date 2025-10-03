"""Setup configuration for HFT Order Book Imbalance Forecasting package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="hft-orderbook-imbalance",
    version="0.1.0",
    author="Mohin Hasin",
    author_email="mohinhasin999@gmail.com",
    description="High-Frequency Order Book Imbalance Forecasting using ML and Computational Statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "polars>=0.18.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "streamlit>=1.24.0",
        "plotly>=5.14.0",
        "kafka-python>=2.0.2",
        "psycopg2-binary>=2.9.6",
        "influxdb-client>=1.36.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "bayesian": [
            "pymc>=5.6.0",
            "arviz>=0.15.0",
        ],
        "tensorflow": [
            "tensorflow>=2.13.0",
            "keras>=2.13.0",
        ],
        "airflow": [
            "apache-airflow>=2.6.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-storage>=2.10.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pymc>=5.6.0",
            "arviz>=0.15.0",
            "tensorflow>=2.13.0",
            "apache-airflow>=2.6.0",
            "boto3>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hft-ingest=ingestion.cli:main",
            "hft-train=models.cli:main",
            "hft-backtest=backtesting.cli:main",
            "hft-api=api.prediction_service:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)
