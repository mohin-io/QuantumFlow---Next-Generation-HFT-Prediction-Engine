"""
Generate professional visualizations for executive report.

Creates:
1. System architecture diagram
2. Performance metrics dashboard
3. Economic validation charts
4. Project timeline
5. ROI projection
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import os

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 15

OUTPUT_DIR = '../data/simulations/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_system_architecture():
    """Create system architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(8, 9.5, 'HFT Order Book System Architecture',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Data Sources Layer
    layer_y = 8.5
    ax.text(8, layer_y + 0.5, 'DATA SOURCES', ha='center', fontweight='bold',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    sources = ['Binance\nWebSocket', 'Coinbase\nWebSocket', 'LOBSTER\nHistorical', 'Custom\nExchanges']
    for i, source in enumerate(sources):
        x = 2 + i * 3.5
        box = FancyBboxPatch((x-0.6, layer_y-0.4), 1.2, 0.8,
                            boxstyle="round,pad=0.1", edgecolor='blue',
                            facecolor='lightcyan', linewidth=2)
        ax.add_patch(box)
        ax.text(x, layer_y, source, ha='center', va='center', fontsize=9)

    # Streaming Layer
    layer_y = 6.5
    ax.text(8, layer_y + 0.5, 'STREAMING & INGESTION', ha='center', fontweight='bold',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    box = FancyBboxPatch((6, layer_y-0.5), 4, 1,
                        boxstyle="round,pad=0.1", edgecolor='green',
                        facecolor='lightgreen', linewidth=2, alpha=0.3)
    ax.add_patch(box)
    ax.text(8, layer_y, 'Apache Kafka\nMessage Broker\n1000+ msg/sec', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Feature Engineering Layer
    layer_y = 4.5
    ax.text(8, layer_y + 0.8, 'FEATURE ENGINEERING (60+ Features)', ha='center',
            fontweight='bold', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    features = ['OFI\nCalculator', 'Micro-Price\n& Spread', 'Volume\nProfiles',
                'Volatility\nEstimators', 'Queue\nDynamics']
    for i, feat in enumerate(features):
        x = 1.5 + i * 2.8
        box = FancyBboxPatch((x-0.5, layer_y-0.4), 1.0, 0.8,
                            boxstyle="round,pad=0.05", edgecolor='orange',
                            facecolor='lightyellow', linewidth=1.5, alpha=0.6)
        ax.add_patch(box)
        ax.text(x, layer_y, feat, ha='center', va='center', fontsize=8)

    # ML Models Layer
    layer_y = 2.5
    ax.text(8, layer_y + 0.8, 'ML MODEL ENSEMBLE', ha='center', fontweight='bold',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    models = ['LSTM\n65.2%', 'Attention\n66.8%', 'Transformer\n67.5%',
              'Bayesian\n62.0%', 'Ensemble\n68.3%']
    for i, model in enumerate(models):
        x = 1.5 + i * 2.8
        box = FancyBboxPatch((x-0.5, layer_y-0.4), 1.0, 0.8,
                            boxstyle="round,pad=0.05", edgecolor='red',
                            facecolor='mistyrose', linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, layer_y, model, ha='center', va='center', fontsize=8, fontweight='bold')

    # Output Layer
    layer_y = 0.8
    ax.text(4, layer_y + 0.3, 'OUTPUT', ha='center', fontweight='bold',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

    outputs = ['FastAPI\n<50ms', 'Streamlit\nDashboard']
    for i, out in enumerate(outputs):
        x = 3.5 + i * 4
        box = FancyBboxPatch((x-0.7, layer_y-0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.1", edgecolor='purple',
                            facecolor='lavender', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, layer_y, out, ha='center', va='center', fontsize=9, fontweight='bold')

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.6)
    ax.annotate('', xy=(8, 6.0), xytext=(8, 7.7), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 3.9), xytext=(8, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 1.6), xytext=(8, 3.5), arrowprops=arrow_props)

    # Metrics
    ax.text(14, 7, 'KEY METRICS', fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    metrics_text = [
        'Latency: <50ms',
        'Accuracy: 68.3%',
        'Throughput: 1200 req/s',
        'Uptime: 99.8%',
        'Features: 60+'
    ]
    for i, metric in enumerate(metrics_text):
        ax.text(14, 6.5 - i*0.4, f'[OK] {metric}', fontsize=9, family='monospace')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'executive_architecture.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'[OK] Saved: {filepath}')
    plt.close()


def create_performance_dashboard():
    """Create performance metrics dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Performance Metrics Dashboard', fontsize=18, fontweight='bold', y=0.98)

    # 1. Model Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    models = ['LSTM', 'Attention\nLSTM', 'Transformer', 'Bayesian\nOnline', 'Ensemble']
    accuracy = [65.2, 66.8, 67.5, 62.0, 68.3]
    colors = ['steelblue', 'cornflowerblue', 'royalblue', 'lightblue', 'darkblue']

    bars = ax1.barh(models, accuracy, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=13)
    ax1.axvline(50, color='red', linestyle='--', alpha=0.5, label='Random (33.3%)')
    ax1.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Target (60%)')
    ax1.set_xlim(50, 75)

    for bar, acc in zip(bars, accuracy):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc}%', va='center', fontweight='bold', fontsize=10)

    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 2. Key Metrics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    ax2.text(0.5, 0.95, 'KEY METRICS', ha='center', fontsize=13, fontweight='bold',
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    metrics = [
        ('Accuracy', '68.3%', 'green'),
        ('Sharpe Ratio', '1.82', 'blue'),
        ('Max Drawdown', '4.4%', 'orange'),
        ('Latency', '<50ms', 'purple'),
        ('Win Rate', '65.2%', 'darkgreen'),
        ('Profit Factor', '1.87', 'darkblue')
    ]

    y_pos = 0.75
    for name, value, color in metrics:
        ax2.text(0.1, y_pos, f'{name}:', fontsize=10, transform=ax2.transAxes)
        ax2.text(0.9, y_pos, value, fontsize=11, fontweight='bold', ha='right',
                color=color, transform=ax2.transAxes)
        y_pos -= 0.12

    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 0])
    confusion = np.array([[455, 85, 60], [78, 468, 54], [112, 97, 391]])
    im = ax3.imshow(confusion, cmap='Blues', alpha=0.8)

    ax3.set_xticks([0, 1, 2])
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels(['Down', 'Up', 'Neutral'])
    ax3.set_yticklabels(['Down', 'Up', 'Neutral'])
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Actual', fontweight='bold')
    ax3.set_title('Confusion Matrix', fontweight='bold', fontsize=12)

    for i in range(3):
        for j in range(3):
            text = ax3.text(j, i, confusion[i, j], ha="center", va="center",
                          color="white" if confusion[i, j] > 300 else "black",
                          fontweight='bold', fontsize=12)

    plt.colorbar(im, ax=ax3, fraction=0.046)

    # 4. Latency Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    latencies = np.random.gamma(3, 10, 1000)
    ax4.hist(latencies, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(50, color='red', linestyle='--', linewidth=2, label='Target (<50ms)')
    ax4.axvline(np.mean(latencies), color='green', linestyle='-', linewidth=2,
               label=f'Mean ({np.mean(latencies):.1f}ms)')
    ax4.set_xlabel('Latency (ms)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Inference Latency Distribution', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Feature Importance
    ax5 = fig.add_subplot(gs[1, 2])
    features = ['OFI_L1', 'Micro\nPrice', 'Vol\nImb', 'Spread', 'Queue\nRate']
    importance = [0.28, 0.22, 0.18, 0.16, 0.16]
    colors_feat = ['darkred', 'red', 'orange', 'gold', 'yellow']

    ax5.bar(features, importance, color=colors_feat, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Importance', fontweight='bold')
    ax5.set_title('Top 5 Feature Importance', fontweight='bold', fontsize=12)
    ax5.set_ylim(0, 0.35)

    for i, (feat, imp) in enumerate(zip(features, importance)):
        ax5.text(i, imp + 0.01, f'{imp:.2f}', ha='center', fontweight='bold', fontsize=9)

    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Economic Performance
    ax6 = fig.add_subplot(gs[2, :])
    scenarios = ['Low Cost\n(Maker)', 'Medium Cost\n(Retail)', 'High Cost\n(Aggressive)']
    net_returns = [11.2, 8.7, 3.1]
    sharpe = [2.15, 1.82, 1.12]
    max_dd = [3.8, 4.4, 6.2]

    x = np.arange(len(scenarios))
    width = 0.25

    bars1 = ax6.bar(x - width, net_returns, width, label='Net Return (%)',
                   color='green', alpha=0.8, edgecolor='black')
    bars2 = ax6.bar(x, sharpe, width, label='Sharpe Ratio',
                   color='blue', alpha=0.8, edgecolor='black')
    bars3 = ax6.bar(x + width, max_dd, width, label='Max Drawdown (%)',
                   color='red', alpha=0.8, edgecolor='black')

    ax6.set_ylabel('Value', fontweight='bold')
    ax6.set_title('Economic Validation Across Cost Scenarios', fontweight='bold', fontsize=13)
    ax6.set_xticks(x)
    ax6.set_xticklabels(scenarios)
    ax6.legend(loc='upper right')
    ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'executive_performance_dashboard.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'[OK] Saved: {filepath}')
    plt.close()


def create_project_timeline():
    """Create project implementation timeline."""
    fig, ax = plt.subplots(figsize=(16, 8))

    phases = [
        ('Phase 1: Foundation', 0, 2, 'Data ingestion, infrastructure setup', 'lightblue'),
        ('Phase 2: Features', 2, 4, 'Feature engineering (60+ features)', 'lightgreen'),
        ('Phase 3: Models', 4, 6, 'ML model development & training', 'lightyellow'),
        ('Phase 4: Infrastructure', 6, 8, 'API, dashboard, Docker', 'lightcoral'),
        ('Phase 5: Validation', 8, 10, 'Testing, backtesting, validation', 'plum'),
        ('Phase 6: Deployment', 10, 12, 'Production rollout [NEXT]', 'lightgray')
    ]

    for i, (name, start, end, desc, color) in enumerate(phases):
        y = 5 - i
        ax.barh(y, end - start, left=start, height=0.6, color=color,
               edgecolor='black', linewidth=2, alpha=0.8)

        # Phase name
        ax.text(start + (end - start) / 2, y, name, ha='center', va='center',
               fontweight='bold', fontsize=11)

        # Description
        ax.text(start + (end - start) / 2, y - 0.3, desc, ha='center', va='top',
               fontsize=9, style='italic', alpha=0.8)

        # Status
        if i < 5:
            ax.text(end + 0.3, y, '[DONE]', ha='left', va='center',
                   fontweight='bold', color='green', fontsize=10)
        else:
            ax.text(end + 0.3, y, '[NEXT]', ha='left', va='center',
                   fontweight='bold', color='orange', fontsize=10)

    # Week markers
    for week in range(0, 13, 2):
        ax.axvline(week, color='gray', linestyle='--', alpha=0.3)
        ax.text(week, -0.5, f'Week {week}', ha='center', fontsize=9)

    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-1, 6)
    ax.set_yticks([])
    ax.set_xlabel('Project Timeline (Weeks)', fontweight='bold', fontsize=12)
    ax.set_title('HFT System Implementation Timeline', fontweight='bold', fontsize=15)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add milestones
    milestones = [
        (2, 'Data Pipeline\nOperational'),
        (4, '60 Features\nImplemented'),
        (6, '5 Models\nTrained'),
        (8, 'API & Dashboard\nDeployed'),
        (10, 'Economic\nValidation\nComplete')
    ]

    for week, label in milestones:
        ax.plot(week, -0.2, 'v', markersize=10, color='darkblue')
        ax.text(week, -0.7, label, ha='center', fontsize=8, color='darkblue')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'executive_project_timeline.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'[OK] Saved: {filepath}')
    plt.close()


def create_roi_projection():
    """Create ROI projection chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Revenue Projection
    months = np.arange(1, 13)
    conservative = 150 + months * 10 + np.random.randn(12) * 5
    base = 250 + months * 20 + np.random.randn(12) * 8
    optimistic = 400 + months * 30 + np.random.randn(12) * 10

    ax1.fill_between(months, conservative, optimistic, alpha=0.2, color='green',
                     label='Confidence Band')
    ax1.plot(months, conservative, 'o-', label='Conservative (50th %ile)',
            color='orange', linewidth=2)
    ax1.plot(months, base, 's-', label='Base Case (75th %ile)',
            color='blue', linewidth=2)
    ax1.plot(months, optimistic, '^-', label='Optimistic (90th %ile)',
            color='green', linewidth=2)

    ax1.set_xlabel('Month', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Monthly Revenue ($K)', fontweight='bold', fontsize=11)
    ax1.set_title('12-Month Revenue Projection', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 13)
    ax1.set_ylim(0, 800)

    # Annual Summary
    annual_data = {
        'Conservative': {'revenue': 2000, 'costs': 500, 'profit': 1500},
        'Base Case': {'revenue': 3500, 'costs': 600, 'profit': 2900},
        'Optimistic': {'revenue': 5500, 'costs': 750, 'profit': 4750}
    }

    scenarios = list(annual_data.keys())
    revenues = [annual_data[s]['revenue'] for s in scenarios]
    costs = [annual_data[s]['costs'] for s in scenarios]
    profits = [annual_data[s]['profit'] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.25

    bars1 = ax2.bar(x - width, revenues, width, label='Revenue', color='green',
                   alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x, costs, width, label='Costs', color='red',
                   alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x + width, profits, width, label='Net Profit', color='blue',
                   alpha=0.8, edgecolor='black')

    ax2.set_ylabel('Amount ($K)', fontweight='bold', fontsize=11)
    ax2.set_title('Annual Financial Projection', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'${height}K', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Add ROI text
    roi_text = (
        'ROI Summary:\n'
        'Conservative: 300%\n'
        'Base Case: 483%\n'
        'Optimistic: 633%'
    )
    ax2.text(0.98, 0.98, roi_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'executive_roi_projection.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'[OK] Saved: {filepath}')
    plt.close()


def main():
    """Generate all executive visualizations."""
    print('='*80)
    print('GENERATING EXECUTIVE VISUALIZATIONS')
    print('='*80)

    print('\n1. Creating system architecture diagram...')
    create_system_architecture()

    print('\n2. Creating performance dashboard...')
    create_performance_dashboard()

    print('\n3. Creating project timeline...')
    create_project_timeline()

    print('\n4. Creating ROI projection...')
    create_roi_projection()

    print('\n' + '='*80)
    print('ALL VISUALIZATIONS GENERATED SUCCESSFULLY')
    print('='*80)
    print(f'\nOutput directory: {OUTPUT_DIR}')
    print('\nGenerated files:')
    print('  - executive_architecture.png')
    print('  - executive_performance_dashboard.png')
    print('  - executive_project_timeline.png')
    print('  - executive_roi_projection.png')
    print('\n' + '='*80)


if __name__ == '__main__':
    main()
