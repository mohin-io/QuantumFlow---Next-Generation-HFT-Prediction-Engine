#!/usr/bin/env python
"""
Streamlit Cloud Deployment Helper - Generates deployment URL
"""

import webbrowser
import urllib.parse

# Repository details
GITHUB_USER = "mohin-io"
REPO_NAME = "QuantumFlow---Next-Generation-HFT-Prediction-Engine"
BRANCH = "master"
MAIN_FILE = "run_trading_dashboard.py"

def main():
    print("\n" + "=" * 70)
    print("  STREAMLIT CLOUD DEPLOYMENT")
    print("=" * 70 + "\n")

    print("Repository Details:")
    print(f"  GitHub: {GITHUB_USER}/{REPO_NAME}")
    print(f"  Branch: {BRANCH}")
    print(f"  Main File: {MAIN_FILE}\n")

    # Generate pre-filled deployment URL
    base_url = "https://share.streamlit.io"

    print("=" * 70)
    print("  DEPLOYMENT OPTIONS")
    print("=" * 70 + "\n")

    print("OPTION 1: Direct Streamlit Cloud (Recommended)")
    print("-" * 70)
    print(f"1. Go to: {base_url}")
    print("2. Sign in with GitHub")
    print("3. Click 'New app'")
    print("4. Enter these details:")
    print(f"   - Repository: {GITHUB_USER}/{REPO_NAME}")
    print(f"   - Branch: {BRANCH}")
    print(f"   - Main file path: {MAIN_FILE}")
    print("5. Click 'Deploy!'\n")

    # Repository URL
    repo_url = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"

    print("OPTION 2: Quick Link")
    print("-" * 70)
    print("Click this URL to open Streamlit Cloud:")
    print(f"{base_url}\n")

    print("GitHub Repository:")
    print(f"{repo_url}\n")

    print("=" * 70)
    print("  AFTER DEPLOYMENT")
    print("=" * 70 + "\n")

    print("Your app will be available at a URL like:")
    print("  https://quantumflow-hft-dashboard.streamlit.app")
    print("  (exact URL will be shown after deployment)\n")

    print("Deployment typically takes 2-5 minutes.")
    print("\n" + "=" * 70 + "\n")

    # Ask to open browser
    try:
        choice = input("Open Streamlit Cloud in browser? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print("\nOpening browser...")
            webbrowser.open(base_url)
            print("Browser opened! Sign in and create new app.\n")
        else:
            print(f"\nVisit {base_url} when ready to deploy.\n")
    except:
        print(f"\nVisit {base_url} to deploy your app.\n")

if __name__ == "__main__":
    main()
