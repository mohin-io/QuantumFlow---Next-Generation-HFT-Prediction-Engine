# ESG Analytics Dashboard - User Guide

## Overview

The **ESG Analytics Dashboard** is a comprehensive Streamlit application for evaluating and monitoring corporate Environmental, Social, and Governance (ESG) performance. It provides real-time scoring, risk-return analysis, sentiment monitoring, and scenario planning capabilities.

---

## Features

### 1. 🏢 Company ESG Health Scorecards

**Comprehensive ESG evaluation with visual scorecards for each company.**

#### What You Get:
- **Overall ESG Score** (0-100 scale)
- **ESG Rating** (AAA to B classification)
- **Pillar Breakdown**:
  - 🌱 Environmental (35% weight)
  - 👥 Social (35% weight)
  - ⚖️ Governance (30% weight)

#### Visual Components:
- **Radar Charts**: Multi-dimensional view of all ESG components
- **Progress Bars**: Quick visual assessment of pillar performance
- **Detailed Metrics Table**: Raw data for all key indicators

#### Rating System:
- **AAA (80-100)**: Leader - Best-in-class ESG performance
- **AA (70-79)**: Advanced - Strong ESG practices
- **A (60-69)**: Good - Above-average performance
- **BBB (50-59)**: Average - Meets basic standards
- **BB (40-49)**: Below Average - Needs improvement
- **B (<40)**: Laggard - Significant ESG risks

---

### 2. 📊 Risk-Return Tradeoff Visualization

**Analyze the relationship between ESG performance and financial returns.**

#### Interactive Scatter Plot:
- **X-axis**: Risk (Volatility %)
- **Y-axis**: Expected Return (%)
- **Color**: ESG Score (green = high, red = low)
- **Size**: Market Capitalization

#### Key Insights:
- **ESG Premium**: Higher ESG scores correlate with better risk-adjusted returns
- **Volatility Reduction**: ESG leaders tend to have lower volatility
- **Sharpe Ratio**: Best risk-adjusted performance highlighted
- **Efficient Frontier**: Optimal risk-return combinations

#### Correlation Analysis:
- **ESG vs Return**: Positive correlation demonstrates ESG premium
- **ESG vs Risk**: Negative correlation shows risk mitigation
- **Trendline Analysis**: Statistical relationships with OLS regression

---

### 3. 🚨 Real-Time Sentiment-Driven Alerts

**Automated monitoring system for ESG risks and opportunities.**

#### Alert Categories:

##### 🔴 HIGH Severity
- Environmental violations detected
- High carbon intensity (>2.5 tons/$M revenue)
- Excessive safety incidents (>5 per year)
- Critical governance issues

##### 🟡 MEDIUM Severity
- Below-target employee satisfaction (<70)
- High executive compensation ratios (>200x)
- Social performance concerns

##### 🟢 LOW Severity (Opportunities)
- ESG Leader status (>75 score)
- Investment opportunities
- Positive sustainability trends

#### Alert Details:
- **Timestamp**: When the alert was triggered
- **Company**: Affected organization
- **Category**: Environmental, Social, Governance, or Opportunity
- **Message**: Description of the issue/opportunity
- **Recommended Action**: Specific next steps

#### Filtering Options:
- Filter by severity level
- Filter by category
- Filter by company
- Real-time refresh capability

---

### 4. 🎯 Interactive What-If Simulator

**Explore how ESG metric changes impact overall scores and ratings.**

#### Adjustable Parameters:

##### Environmental Factors:
- **CO2 Emissions**: ±50% adjustment
- **Renewable Energy**: ±30 percentage points
- **Waste Recycling**: ±30 percentage points

##### Social Factors:
- **Employee Satisfaction**: ±20 points
- **Diversity Score**: ±20 points

##### Governance Factors:
- **Female Board Members**: ±20 percentage points

#### Real-Time Impact Analysis:
- **Score Changes**: Immediate recalculation of all scores
- **Rating Changes**: See if adjustments cause rating upgrades/downgrades
- **Pillar Impact**: Breakdown showing which pillar is most affected
- **Visual Comparison**: Side-by-side bar charts of baseline vs scenario

#### Use Cases:
1. **Target Setting**: "What CO2 reduction needed to reach AA rating?"
2. **Investment Planning**: "How much ESG improvement for target return?"
3. **Risk Assessment**: "Impact of potential governance issues?"
4. **Scenario Planning**: "Best combination of improvements?"

---

### 5. 📋 Portfolio Overview

**Aggregate view of all companies in your ESG universe.**

#### Summary Metrics:
- **Portfolio Average ESG Score**
- **Number of Companies Analyzed**
- **ESG Leaders Count** (score ≥70)
- **ESG Laggards Count** (score <50)

#### Visualizations:
- **Holdings Table**: Complete portfolio with color-coded scores
- **Sector Analysis**: Average ESG performance by industry
- **Rating Distribution**: Pie chart of ESG rating breakdown

---

## ESG Scoring Methodology

### Environmental Pillar (35% weight)

#### Components:
1. **CO2 Efficiency** (30%)
   - Metric: Tons of CO2 per $M revenue
   - Lower is better
   - Benchmark: 500 tons/$M

2. **Water Usage** (20%)
   - Metric: Cubic meters per $M revenue
   - Lower is better
   - Benchmark: 1000 m³/$M

3. **Renewable Energy** (25%)
   - Metric: % of total energy from renewables
   - Higher is better
   - Target: 100%

4. **Waste Recycling** (20%)
   - Metric: % of waste recycled
   - Higher is better
   - Target: 100%

5. **Compliance** (5% penalty)
   - Metric: Number of violations
   - Each violation: -10 points

### Social Pillar (35% weight)

#### Components:
1. **Employee Satisfaction** (25%)
   - Metric: 0-100 survey score
   - Target: ≥80

2. **Diversity & Inclusion** (25%)
   - Metric: 0-100 composite score
   - Target: ≥75

3. **Safety** (20%)
   - Metric: Number of incidents
   - Lower is better
   - Each incident: -5 points

4. **Community Investment** (15%)
   - Metric: % of revenue invested in community
   - Higher is better

5. **Supply Chain Ethics** (15%)
   - Metric: 0-100 audit score
   - Target: ≥80

### Governance Pillar (30% weight)

#### Components:
1. **Board Independence** (25%)
   - Metric: % of independent directors
   - Target: ≥75%

2. **Gender Diversity** (20%)
   - Metric: % female board members
   - Target: ≥50% (parity)

3. **Executive Compensation** (20%)
   - Metric: CEO to median employee ratio
   - Lower is better
   - Benchmark: 150x

4. **Audit Quality** (20%)
   - Metric: 0-100 audit score
   - Target: ≥90

5. **Shareholder Rights** (15%)
   - Metric: 0-100 rights score
   - Target: ≥85

---

## Installation & Setup

### Prerequisites
```bash
Python 3.9+
pip
```

### Installation
```bash
# Clone repository
git clone https://github.com/mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine.git
cd hft-order-book-imbalance

# Install dependencies
pip install -r requirements.txt
```

### Launch Dashboard
```bash
# Option 1: Using launcher script
python run_esg_dashboard.py

# Option 2: Direct streamlit command
streamlit run src/esg/esg_dashboard.py --server.port=8502
```

The dashboard will automatically open in your browser at `http://localhost:8502`

---

## Sample Data

The dashboard includes 5 sample companies representing different sectors:

1. **TechCorp Inc.** (Technology)
   - ESG Leader with AAA rating
   - Strong environmental and governance scores

2. **EnergyMax Corp.** (Energy)
   - A rating, moderate performance
   - High emissions but improving

3. **GreenBank Ltd.** (Financial Services)
   - Top performer with AAA rating
   - Best-in-class across all pillars

4. **MineralCo Resources** (Mining)
   - BBB rating, average performance
   - Significant environmental challenges

5. **ConsumerGoods Global** (Consumer Goods)
   - AA rating, good performance
   - Balanced across all three pillars

---

## Use Cases

### For Investment Managers:
- **Portfolio Construction**: Build ESG-optimized portfolios
- **Risk Assessment**: Identify ESG-related risks in holdings
- **Client Reporting**: Generate ESG scorecards for clients
- **Engagement Priorities**: Target companies for ESG improvement

### For Corporate Sustainability Teams:
- **Benchmarking**: Compare performance against peers
- **Target Setting**: Use what-if simulator to set realistic goals
- **Progress Tracking**: Monitor ESG improvements over time
- **Stakeholder Communication**: Visual reports for annual disclosures

### For Risk Management:
- **Alert Monitoring**: Real-time notification of ESG risks
- **Scenario Analysis**: Stress-test ESG under various conditions
- **Compliance Tracking**: Monitor violations and incidents
- **Reputation Risk**: Identify potential ESG controversies

### For Analysts:
- **Research**: Comprehensive ESG data analysis
- **Correlation Studies**: ESG vs financial performance
- **Sector Analysis**: Industry-specific ESG trends
- **Rating Validation**: Verify and cross-check ESG ratings

---

## Technical Architecture

### Data Models
```python
ESGMetrics (dataclass)
├── Environmental Metrics (5 fields)
├── Social Metrics (5 fields)
├── Governance Metrics (5 fields)
└── Financial Metrics (2 fields)
```

### Scoring Engine
```python
ESGScorer
├── calculate_environmental_score()
├── calculate_social_score()
├── calculate_governance_score()
└── calculate_overall_score()
```

### Visualization Framework
- **Plotly**: Interactive charts and graphs
- **Streamlit**: Dashboard UI and layout
- **Pandas**: Data manipulation and analysis

---

## Best Practices

### Data Quality:
1. Ensure all metrics are up-to-date (quarterly updates recommended)
2. Validate data sources and methodologies
3. Document any data adjustments or normalizations
4. Maintain audit trail of ESG data changes

### Scoring Interpretation:
1. Compare companies within same sector for meaningful insights
2. Consider trends over time, not just absolute scores
3. Investigate component scores, not just overall rating
4. Combine ESG scores with financial analysis

### Alert Management:
1. Review alerts daily during market hours
2. Prioritize HIGH severity alerts
3. Document actions taken for each alert
4. Adjust alert thresholds based on materiality

### Scenario Planning:
1. Test multiple scenarios, not just optimistic cases
2. Consider realistic improvement timelines
3. Factor in costs of ESG improvements
4. Validate scenarios with subject matter experts

---

## Customization

### Adding New Companies:
```python
new_company = ESGMetrics(
    company="NewCorp",
    sector="Technology",
    # ... add all required metrics
)
```

### Adjusting Weights:
```python
scorer.weights = {
    'environmental': 0.40,  # Increase environmental weight
    'social': 0.30,
    'governance': 0.30
}
```

### Custom Alert Rules:
Modify `create_sentiment_alerts()` function to add custom logic.

---

## Troubleshooting

### Dashboard won't start:
```bash
# Check streamlit installation
pip install --upgrade streamlit

# Check port availability
netstat -an | findstr 8502
```

### Data not displaying:
- Verify sample data generation in `esg_models.py`
- Check browser console for JavaScript errors
- Clear browser cache and reload

### Slow performance:
- Reduce number of companies analyzed
- Simplify visualizations (fewer data points)
- Use caching for expensive calculations

---

## Future Enhancements

### Planned Features:
- [ ] Historical ESG score tracking
- [ ] Peer comparison tool
- [ ] PDF report generation
- [ ] Data import from CSV/Excel
- [ ] Integration with ESG data providers (MSCI, Sustainalytics)
- [ ] Custom scoring methodology builder
- [ ] Multi-portfolio comparison
- [ ] ESG forecasting models

---

## References

### ESG Standards:
- **GRI (Global Reporting Initiative)**: Sustainability reporting framework
- **SASB (Sustainability Accounting Standards Board)**: Industry-specific standards
- **TCFD (Task Force on Climate-related Financial Disclosures)**: Climate risk reporting
- **UN SDGs (Sustainable Development Goals)**: Global sustainability framework

### Academic Research:
- Friede, G., Busch, T., & Bassen, A. (2015). "ESG and financial performance: aggregated evidence from more than 2000 empirical studies"
- Khan, M., Serafeim, G., & Yoon, A. (2016). "Corporate Sustainability: First Evidence on Materiality"
- Eccles, R. G., & Serafeim, G. (2013). "The Performance Frontier: Innovating for a Sustainable Strategy"

---

## Support

For questions, issues, or feature requests:
- GitHub Issues: [https://github.com/mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine/issues](https://github.com/mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine/issues)
- Documentation: See `docs/` folder

---

## License

MIT License - See LICENSE file for details

---

**Last Updated:** October 2025
**Version:** 1.0.0
