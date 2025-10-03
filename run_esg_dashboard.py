"""
Launcher script for ESG Analytics Dashboard.

Usage:
    python run_esg_dashboard.py
"""

import subprocess
import sys
import os

def main():
    """Launch the ESG dashboard."""

    print("="*80)
    print("LAUNCHING ESG ANALYTICS DASHBOARD")
    print("="*80)
    print("\nStarting Streamlit server...")
    print("Dashboard will open in your browser automatically.")
    print("\nPress Ctrl+C to stop the server.")
    print("="*80)
    print()

    # Path to dashboard
    dashboard_path = os.path.join('src', 'esg', 'esg_dashboard.py')

    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            dashboard_path,
            '--server.port=8502',
            '--server.headless=false',
            '--theme.primaryColor=#4CAF50',
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
