"""
Launcher script for HFT Live Trading Dashboard.

Real-time market data and trading signals.

Usage:
    python run_hft_live_dashboard.py
"""

import subprocess
import sys
import os

def main():
    """Launch the HFT live dashboard."""

    print("="*80)
    print("LAUNCHING HFT LIVE TRADING DASHBOARD")
    print("="*80)
    print("\n📊 Features:")
    print("  - Live order book visualization from Binance/Coinbase")
    print("  - Real-time signal generation")
    print("  - Execution simulation with live prices")
    print("  - Performance tracking")
    print("  - Cross-exchange arbitrage detection")
    print("\n" + "="*80)
    print("\nStarting Streamlit server...")
    print("Dashboard will open in your browser automatically.")
    print("\nPress Ctrl+C to stop the server.")
    print("="*80)
    print()

    # Path to dashboard
    dashboard_path = os.path.join('src', 'visualization', 'hft_live_dashboard.py')

    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            dashboard_path,
            '--server.port=8503',
            '--server.headless=false',
            '--theme.primaryColor=#1E88E5',
            '--theme.backgroundColor=#FFFFFF',
            '--theme.secondaryBackgroundColor=#F0F2F6',
            '--theme.textColor=#262730'
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except Exception as e:
        print(f"\nError launching dashboard: {e}")
        print("\nTry running directly:")
        print(f"  streamlit run {dashboard_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
