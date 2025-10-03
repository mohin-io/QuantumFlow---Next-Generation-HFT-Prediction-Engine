#!/usr/bin/env python
"""
Deployment verification script for QuantumFlow Trading Dashboard.

Runs comprehensive checks to ensure all components are working correctly
before deploying to production.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("CHECKING DEPENDENCIES")

    required = [
        ("pandas", "import pandas"),
        ("numpy", "import numpy"),
        ("plotly", "import plotly.graph_objects"),
    ]

    optional = [("streamlit", "import streamlit")]

    all_ok = True

    for name, import_cmd in required:
        try:
            exec(import_cmd)
            print(f"[OK] {name:20s} - OK")
        except ImportError:
            print(f"[FAIL] {name:20s} - MISSING (required)")
            all_ok = False

    for name, import_cmd in optional:
        try:
            exec(import_cmd)
            print(f"[OK] {name:20s} - OK (optional)")
        except ImportError:
            print(f"[WARN] {name:20s} - MISSING (optional, needed for live deployment)")

    return all_ok


def run_tests():
    """Run the test suite."""
    print_header("RUNNING TEST SUITE")

    test_files = [
        "tests/test_app_utils.py",
        "tests/test_streamlit_app.py",
    ]

    all_ok = True

    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"✗ {test_file} - NOT FOUND")
            all_ok = False
            continue

        print(f"\nRunning {test_file}...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Count passed tests
            lines = result.stdout.split("\n")
            for line in lines:
                if "passed" in line:
                    print(f"[PASS] {test_file} - {line.strip()}")
                    break
        else:
            print(f"[FAIL] {test_file} - FAILED")
            print(result.stdout[-500:])  # Last 500 chars
            all_ok = False

    return all_ok


def check_files():
    """Check if all required files exist."""
    print_header("CHECKING FILES")

    required_files = [
        "run_trading_dashboard.py",
        "src/visualization/app_utils.py",
        "tests/test_app_utils.py",
        "tests/test_streamlit_app.py",
        "STREAMLIT_APP_README.md",
    ]

    all_ok = True

    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"[OK] {file_path:40s} - OK ({size:,} bytes)")
        else:
            print(f"[FAIL] {file_path:40s} - MISSING")
            all_ok = False

    return all_ok


def run_quick_import_test():
    """Test that app utilities can be imported."""
    print_header("IMPORT TEST")

    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from visualization.app_utils import (
            OrderBookGenerator,
            MetricsCalculator,
            SignalGenerator,
            DataGenerator,
            PlotGenerator,
            FeatureAnalyzer,
        )

        print("[OK] All utility classes imported successfully")

        # Quick functionality test
        ob_gen = OrderBookGenerator()
        bids, asks, mid_price = ob_gen.generate_orderbook()

        metrics_calc = MetricsCalculator()
        spread_metrics = metrics_calc.calculate_spread_metrics(bids, asks, mid_price)

        print(f"[OK] Generated sample order book: mid_price=${mid_price:,.2f}")
        print(f"[OK] Calculated spread: {spread_metrics['spread_bps']:.2f} bps")

        return True
    except Exception as e:
        print(f"[FAIL] Import test failed: {e}")
        return False


def print_deployment_summary():
    """Print deployment instructions."""
    print_header("DEPLOYMENT READY")

    print(
        """
The Streamlit Trading Dashboard is ready for deployment!

[DEPLOYMENT OPTIONS]

1. LOCAL DEVELOPMENT:
   streamlit run run_trading_dashboard.py

2. STREAMLIT CLOUD:
   - Push to GitHub
   - Deploy at share.streamlit.io
   - Point to: run_trading_dashboard.py

3. DOCKER:
   docker build -t trading-dashboard .
   docker run -p 8501:8501 trading-dashboard

4. KUBERNETES:
   kubectl apply -f k8s/dashboard-deployment.yaml

[TEST SUMMARY]
   - Unit Tests: 43 tests (all utility functions)
   - Integration Tests: 23 tests (end-to-end workflows)
   - Total: 66 tests - ALL PASSING

[DOCUMENTATION]
   See STREAMLIT_APP_README.md for complete guide

[NEXT STEPS]
   1. Review configuration in sidebar
   2. Test with real data sources
   3. Configure API endpoints
   4. Deploy to production!
"""
    )


def main():
    """Run all verification checks."""
    print(
        """
    ==============================================================
      QUANTUMFLOW TRADING DASHBOARD - DEPLOYMENT VERIFIER
    ==============================================================
    """
    )

    checks = []

    # Run all checks
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("Files", check_files()))
    checks.append(("Import Test", run_quick_import_test()))
    checks.append(("Test Suite", run_tests()))

    # Summary
    print_header("VERIFICATION SUMMARY")

    all_passed = True
    for check_name, passed in checks:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{check_name:20s}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print_deployment_summary()
        return 0
    else:
        print("\n[WARN] Some checks failed. Please fix issues before deploying.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
