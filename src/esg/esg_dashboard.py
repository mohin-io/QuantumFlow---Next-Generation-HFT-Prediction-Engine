"""
Comprehensive ESG Dashboard with Streamlit

Features:
1. Company ESG Health Scorecards
2. Risk-Return Tradeoff Visualization
3. Real-time Sentiment-Driven Alerts
4. Interactive What-If Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esg.esg_models import ESGMetrics, ESGScorer, generate_sample_companies

# Page configuration
st.set_page_config(
    page_title="ESG Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    .alert-medium {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    .alert-low {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
if "companies" not in st.session_state:
    st.session_state.companies = generate_sample_companies()
    st.session_state.scorer = ESGScorer()
    st.session_state.alerts = []
    st.session_state.last_update = datetime.now()


def create_esg_scorecard(company: ESGMetrics):
    """Create detailed ESG scorecard for a company."""

    scorer = st.session_state.scorer
    result = scorer.calculate_overall_score(company)

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"## {company.company}")
        st.markdown(
            f"**Sector:** {company.sector} | **Market Cap:** ${company.market_cap:,.0f}M"
        )

    with col2:
        st.metric("Overall ESG Score", f"{result['overall_score']:.1f}")

    with col3:
        rating_color = scorer.get_rating_color(result["rating"])
        st.markdown(
            f"<div style='background:{rating_color}; padding:1rem; border-radius:8px; text-align:center;'>"
            f"<h2 style='color:white; margin:0;'>{result['rating']}</h2>"
            f"<p style='color:white; margin:0;'>{result['rating_description']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Pillar scores
    st.markdown("### ESG Pillar Scores")
    col1, col2, col3 = st.columns(3)

    pillars = [
        ("Environmental", result["pillar_scores"]["environmental"], "🌱"),
        ("Social", result["pillar_scores"]["social"], "👥"),
        ("Governance", result["pillar_scores"]["governance"], "⚖️"),
    ]

    for col, (name, score, emoji) in zip([col1, col2, col3], pillars):
        with col:
            st.markdown(f"#### {emoji} {name}")
            st.progress(score / 100)
            st.markdown(f"**Score: {score:.1f}/100**")

    # Radar chart
    st.markdown("### Detailed Component Analysis")

    fig = go.Figure()

    # Environmental components
    env_comp = result["detailed_components"]["environmental"]
    categories = [
        "CO2 Efficiency",
        "Water Usage",
        "Renewable Energy",
        "Waste Recycling",
        "Compliance",
    ]
    values = [
        env_comp["co2_score"],
        env_comp["water_score"],
        env_comp["renewable_score"],
        env_comp["waste_score"],
        100 - env_comp["violation_penalty"],
    ]

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Environmental",
            line_color="green",
        )
    )

    # Social components
    social_comp = result["detailed_components"]["social"]
    categories_social = [
        "Employee Satisfaction",
        "Diversity",
        "Safety",
        "Community Investment",
        "Supply Chain Ethics",
    ]
    values_social = [
        social_comp["satisfaction_score"],
        social_comp["diversity_score"],
        social_comp["safety_score"],
        social_comp["community_score"],
        social_comp["ethics_score"],
    ]

    fig.add_trace(
        go.Scatterpolar(
            r=values_social,
            theta=categories_social,
            fill="toself",
            name="Social",
            line_color="blue",
        )
    )

    # Governance components
    gov_comp = result["detailed_components"]["governance"]
    categories_gov = [
        "Board Independence",
        "Gender Diversity",
        "Fair Compensation",
        "Audit Quality",
        "Shareholder Rights",
    ]
    values_gov = [
        gov_comp["independence_score"],
        gov_comp["diversity_score"],
        gov_comp["comp_score"],
        gov_comp["audit_score"],
        gov_comp["shareholder_score"],
    ]

    fig.add_trace(
        go.Scatterpolar(
            r=values_gov,
            theta=categories_gov,
            fill="toself",
            name="Governance",
            line_color="purple",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key metrics table
    st.markdown("### Key Metrics")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "CO2 Emissions (tons/year)",
                "Renewable Energy %",
                "Employee Satisfaction",
                "Board Independence %",
                "Female Board Members %",
            ],
            "Value": [
                f"{company.co2_emissions:,.0f}",
                f"{company.renewable_energy_pct:.1f}%",
                f"{company.employee_satisfaction:.1f}",
                f"{company.board_independence:.1f}%",
                f"{company.female_board_members:.1f}%",
            ],
        }
    )

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def create_risk_return_viz():
    """Create risk-return tradeoff visualization."""

    st.markdown("## 📊 Risk-Return Tradeoff Analysis")

    companies = st.session_state.companies
    scorer = st.session_state.scorer

    # Calculate scores and synthetic risk/return
    data = []
    for company in companies:
        result = scorer.calculate_overall_score(company)

        # Synthetic expected return (ESG leaders tend to have better long-term returns)
        base_return = 8
        esg_premium = (
            (result["overall_score"] - 50) / 10 * 2
        )  # 2% per 10 points above 50
        expected_return = base_return + esg_premium + np.random.randn() * 1

        # Synthetic volatility (ESG leaders tend to have lower risk)
        base_vol = 20
        esg_risk_reduction = (
            (result["overall_score"] - 50) / 10 * 1.5
        )  # -1.5% vol per 10 points
        volatility = max(8, base_vol - esg_risk_reduction + np.random.randn() * 2)

        # Sharpe ratio
        sharpe = expected_return / volatility

        data.append(
            {
                "company": company.company,
                "sector": company.sector,
                "esg_score": result["overall_score"],
                "rating": result["rating"],
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "market_cap": company.market_cap,
            }
        )

    df = pd.DataFrame(data)

    # Scatter plot
    fig = px.scatter(
        df,
        x="volatility",
        y="expected_return",
        size="market_cap",
        color="esg_score",
        hover_data=["company", "sector", "rating", "sharpe_ratio"],
        labels={
            "volatility": "Risk (Volatility %)",
            "expected_return": "Expected Return (%)",
            "esg_score": "ESG Score",
            "market_cap": "Market Cap ($M)",
        },
        title="Risk-Return Profile by ESG Score",
        color_continuous_scale="RdYlGn",
        height=600,
    )

    # Add efficient frontier approximation
    vol_range = np.linspace(df["volatility"].min(), df["volatility"].max(), 100)
    efficient_return = (
        df["expected_return"].max() - 0.01 * (vol_range - df["volatility"].min()) ** 1.5
    )

    fig.add_trace(
        go.Scatter(
            x=vol_range,
            y=efficient_return,
            mode="lines",
            name="Efficient Frontier (Approx)",
            line=dict(color="red", dash="dash", width=2),
        )
    )

    fig.update_layout(
        xaxis_title="Risk (Volatility %)",
        yaxis_title="Expected Return (%)",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_sharpe = df.loc[df["sharpe_ratio"].idxmax()]
        st.metric(
            "Best Sharpe Ratio",
            f"{best_sharpe['sharpe_ratio']:.2f}",
            delta=best_sharpe["company"],
        )

    with col2:
        highest_esg = df.loc[df["esg_score"].idxmax()]
        st.metric(
            "Highest ESG Score",
            f"{highest_esg['esg_score']:.1f}",
            delta=highest_esg["company"],
        )

    with col3:
        avg_return = df["expected_return"].mean()
        st.metric("Average Expected Return", f"{avg_return:.1f}%")

    with col4:
        avg_vol = df["volatility"].mean()
        st.metric("Average Volatility", f"{avg_vol:.1f}%")

    # Correlation analysis
    st.markdown("### ESG Score Impact on Risk-Return")

    col1, col2 = st.columns(2)

    with col1:
        # ESG vs Return
        fig1 = px.scatter(
            df,
            x="esg_score",
            y="expected_return",
            trendline="ols",
            labels={"esg_score": "ESG Score", "expected_return": "Expected Return (%)"},
            title="ESG Score vs Expected Return",
            height=400,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # ESG vs Risk
        fig2 = px.scatter(
            df,
            x="esg_score",
            y="volatility",
            trendline="ols",
            labels={"esg_score": "ESG Score", "volatility": "Volatility (%)"},
            title="ESG Score vs Risk",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)


def create_sentiment_alerts():
    """Create real-time sentiment-driven alerts system."""

    st.markdown("## 🚨 Real-Time ESG Sentiment Alerts")

    # Simulate real-time alerts
    if st.button("🔄 Refresh Alerts", type="primary"):
        st.session_state.last_update = datetime.now()

        # Generate synthetic alerts
        companies = st.session_state.companies
        scorer = st.session_state.scorer

        alerts = []

        for company in companies:
            result = scorer.calculate_overall_score(company)

            # Environmental alerts
            if company.co2_emissions / company.revenue > 2.5:
                alerts.append(
                    {
                        "timestamp": datetime.now(),
                        "company": company.company,
                        "severity": "HIGH",
                        "category": "Environmental",
                        "message": f"High carbon intensity detected: {company.co2_emissions/company.revenue:.2f} tons/$M revenue",
                        "action": "Review emissions reduction targets",
                    }
                )

            if company.environmental_violations > 0:
                alerts.append(
                    {
                        "timestamp": datetime.now(),
                        "company": company.company,
                        "severity": "HIGH",
                        "category": "Environmental",
                        "message": f"{company.environmental_violations} environmental violations reported",
                        "action": "Immediate compliance review required",
                    }
                )

            # Social alerts
            if company.employee_satisfaction < 70:
                alerts.append(
                    {
                        "timestamp": datetime.now(),
                        "company": company.company,
                        "severity": "MEDIUM",
                        "category": "Social",
                        "message": f"Employee satisfaction below target: {company.employee_satisfaction:.0f}/100",
                        "action": "Employee engagement survey needed",
                    }
                )

            if company.safety_incidents > 5:
                alerts.append(
                    {
                        "timestamp": datetime.now(),
                        "company": company.company,
                        "severity": "HIGH",
                        "category": "Social",
                        "message": f"{company.safety_incidents} safety incidents reported",
                        "action": "Safety protocol review required",
                    }
                )

            # Governance alerts
            if company.executive_compensation_ratio > 200:
                alerts.append(
                    {
                        "timestamp": datetime.now(),
                        "company": company.company,
                        "severity": "MEDIUM",
                        "category": "Governance",
                        "message": f"High CEO pay ratio: {company.executive_compensation_ratio:.0f}x median employee",
                        "action": "Compensation committee review",
                    }
                )

            # Positive alerts (opportunities)
            if result["overall_score"] > 75:
                alerts.append(
                    {
                        "timestamp": datetime.now(),
                        "company": company.company,
                        "severity": "LOW",
                        "category": "Opportunity",
                        "message": f"ESG Leader: {result['rating']} rating ({result['overall_score']:.1f} score)",
                        "action": "Consider for sustainable investment portfolio",
                    }
                )

        st.session_state.alerts = alerts
        st.rerun()

    st.markdown(
        f"**Last Updated:** {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
        )

    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            ["Environmental", "Social", "Governance", "Opportunity"],
            default=["Environmental", "Social", "Governance", "Opportunity"],
        )

    with col3:
        company_filter = st.multiselect(
            "Filter by Company",
            [c.company for c in st.session_state.companies],
            default=[c.company for c in st.session_state.companies],
        )

    # Display alerts
    filtered_alerts = [
        a
        for a in st.session_state.alerts
        if a["severity"] in severity_filter
        and a["category"] in category_filter
        and a["company"] in company_filter
    ]

    if not filtered_alerts:
        st.info("No alerts match the current filters.")
    else:
        st.markdown(f"### {len(filtered_alerts)} Active Alerts")

        for alert in sorted(
            filtered_alerts,
            key=lambda x: (x["severity"] != "HIGH", x["timestamp"]),
            reverse=True,
        ):
            severity_class = f"alert-{alert['severity'].lower()}"

            severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                alert["severity"], "⚪"
            )

            st.markdown(
                f"""
                <div class="alert-box {severity_class}">
                    <strong>{severity_icon} {alert['severity']}</strong> |
                    {alert['category']} |
                    <strong>{alert['company']}</strong><br/>
                    📋 {alert['message']}<br/>
                    💡 <em>Recommended Action: {alert['action']}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )


def create_whatif_simulator():
    """Create interactive what-if simulator."""

    st.markdown("## 🎯 Interactive What-If Simulator")

    st.markdown(
        """
    Explore how changes in ESG metrics impact overall scores and ratings.
    Adjust the sliders below to see real-time impact.
    """
    )

    # Select company
    companies = st.session_state.companies
    company_names = [c.company for c in companies]

    selected_company_name = st.selectbox("Select Company", company_names, index=0)

    company = next(c for c in companies if c.company == selected_company_name)
    scorer = st.session_state.scorer

    # Get baseline score
    baseline_result = scorer.calculate_overall_score(company)

    # Display baseline
    st.markdown("### Baseline Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current ESG Score", f"{baseline_result['overall_score']:.1f}")
    with col2:
        st.metric("Current Rating", baseline_result["rating"])
    with col3:
        st.metric("E Score", f"{baseline_result['pillar_scores']['environmental']:.1f}")
    with col4:
        st.metric("S Score", f"{baseline_result['pillar_scores']['social']:.1f}")

    st.markdown("---")
    st.markdown("### What-If Scenario Builder")

    # Create sliders for key metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌱 Environmental Adjustments")

        co2_change = st.slider(
            "CO2 Emissions Change (%)",
            -50,
            50,
            0,
            5,
            help="Negative values represent reductions",
        )

        renewable_change = st.slider(
            "Renewable Energy Change (percentage points)", -30, 30, 0, 5
        )

        waste_change = st.slider(
            "Waste Recycling Change (percentage points)", -30, 30, 0, 5
        )

    with col2:
        st.markdown("#### 👥 Social & Governance Adjustments")

        satisfaction_change = st.slider(
            "Employee Satisfaction Change (points)", -20, 20, 0, 2
        )

        diversity_change = st.slider("Diversity Score Change (points)", -20, 20, 0, 2)

        board_diversity_change = st.slider(
            "Female Board Members Change (percentage points)", -20, 20, 0, 2
        )

    # Apply changes
    modified_company = ESGMetrics(
        company=company.company,
        sector=company.sector,
        revenue=company.revenue,
        market_cap=company.market_cap,
        co2_emissions=company.co2_emissions * (1 + co2_change / 100),
        water_usage=company.water_usage,
        renewable_energy_pct=min(
            100, max(0, company.renewable_energy_pct + renewable_change)
        ),
        waste_recycled_pct=min(100, max(0, company.waste_recycled_pct + waste_change)),
        environmental_violations=company.environmental_violations,
        employee_satisfaction=min(
            100, max(0, company.employee_satisfaction + satisfaction_change)
        ),
        diversity_score=min(100, max(0, company.diversity_score + diversity_change)),
        safety_incidents=company.safety_incidents,
        community_investment=company.community_investment,
        supply_chain_ethics_score=company.supply_chain_ethics_score,
        board_independence=company.board_independence,
        female_board_members=min(
            100, max(0, company.female_board_members + board_diversity_change)
        ),
        executive_compensation_ratio=company.executive_compensation_ratio,
        audit_quality_score=company.audit_quality_score,
        shareholder_rights_score=company.shareholder_rights_score,
    )

    # Calculate new score
    modified_result = scorer.calculate_overall_score(modified_company)

    # Display impact
    st.markdown("---")
    st.markdown("### 📈 Impact Analysis")

    col1, col2, col3, col4 = st.columns(4)

    score_delta = modified_result["overall_score"] - baseline_result["overall_score"]
    env_delta = (
        modified_result["pillar_scores"]["environmental"]
        - baseline_result["pillar_scores"]["environmental"]
    )
    social_delta = (
        modified_result["pillar_scores"]["social"]
        - baseline_result["pillar_scores"]["social"]
    )
    gov_delta = (
        modified_result["pillar_scores"]["governance"]
        - baseline_result["pillar_scores"]["governance"]
    )

    with col1:
        st.metric(
            "New ESG Score",
            f"{modified_result['overall_score']:.1f}",
            f"{score_delta:+.1f}",
            delta_color="normal",
        )

    with col2:
        rating_changed = modified_result["rating"] != baseline_result["rating"]
        st.metric(
            "New Rating",
            modified_result["rating"],
            "Changed!" if rating_changed else "No change",
        )

    with col3:
        st.metric(
            "Environmental",
            f"{modified_result['pillar_scores']['environmental']:.1f}",
            f"{env_delta:+.1f}",
        )

    with col4:
        st.metric(
            "Social",
            f"{modified_result['pillar_scores']['social']:.1f}",
            f"{social_delta:+.1f}",
        )

    # Comparison chart
    st.markdown("### Score Comparison")

    comparison_df = pd.DataFrame(
        {
            "Metric": ["Overall Score", "Environmental", "Social", "Governance"],
            "Baseline": [
                baseline_result["overall_score"],
                baseline_result["pillar_scores"]["environmental"],
                baseline_result["pillar_scores"]["social"],
                baseline_result["pillar_scores"]["governance"],
            ],
            "Modified": [
                modified_result["overall_score"],
                modified_result["pillar_scores"]["environmental"],
                modified_result["pillar_scores"]["social"],
                modified_result["pillar_scores"]["governance"],
            ],
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=comparison_df["Metric"],
            y=comparison_df["Baseline"],
            marker_color="lightblue",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Modified Scenario",
            x=comparison_df["Metric"],
            y=comparison_df["Modified"],
            marker_color="darkblue",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Baseline vs Modified Scenario",
        yaxis_title="Score",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.markdown("### 💡 Key Insights")

    if score_delta > 5:
        st.success(f"✅ Significant improvement: +{score_delta:.1f} points overall")
    elif score_delta > 0:
        st.info(f"📊 Modest improvement: +{score_delta:.1f} points overall")
    elif score_delta < -5:
        st.error(f"⚠️ Significant decline: {score_delta:.1f} points overall")
    elif score_delta < 0:
        st.warning(f"⚠️ Slight decline: {score_delta:.1f} points overall")
    else:
        st.info("No change in overall score")

    if rating_changed:
        st.success(
            f"🎯 Rating changed: {baseline_result['rating']} → {modified_result['rating']}"
        )


# Main application
def main():
    """Main dashboard application."""

    # Header
    st.markdown(
        '<div class="main-header">🌍 ESG Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/300x100/4CAF50/FFFFFF?text=ESG+Analytics",
            use_column_width=True,
        )

        st.markdown("## Navigation")

        page = st.radio(
            "Select View",
            [
                "🏢 Company Scorecards",
                "📊 Risk-Return Analysis",
                "🚨 Sentiment Alerts",
                "🎯 What-If Simulator",
                "📋 Portfolio Overview",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            """
        Comprehensive ESG analytics platform for evaluating corporate sustainability performance.

        **Features:**
        - Real-time ESG scoring
        - Risk-return optimization
        - Sentiment monitoring
        - Scenario analysis
        """
        )

        st.markdown("---")
        st.markdown("**Data as of:** " + datetime.now().strftime("%Y-%m-%d"))

    # Route to selected page
    if page == "🏢 Company Scorecards":
        st.markdown("# Company ESG Health Scorecards")

        # Company selector
        companies = st.session_state.companies
        company_names = [c.company for c in companies]

        tabs = st.tabs(company_names)

        for tab, company in zip(tabs, companies):
            with tab:
                create_esg_scorecard(company)

    elif page == "📊 Risk-Return Analysis":
        create_risk_return_viz()

    elif page == "🚨 Sentiment Alerts":
        create_sentiment_alerts()

    elif page == "🎯 What-If Simulator":
        create_whatif_simulator()

    elif page == "📋 Portfolio Overview":
        st.markdown("## Portfolio ESG Overview")

        companies = st.session_state.companies
        scorer = st.session_state.scorer

        # Calculate all scores
        portfolio_data = []
        for company in companies:
            result = scorer.calculate_overall_score(company)
            portfolio_data.append(
                {
                    "Company": company.company,
                    "Sector": company.sector,
                    "ESG Score": result["overall_score"],
                    "Rating": result["rating"],
                    "E": result["pillar_scores"]["environmental"],
                    "S": result["pillar_scores"]["social"],
                    "G": result["pillar_scores"]["governance"],
                    "Market Cap ($M)": company.market_cap,
                }
            )

        df = pd.DataFrame(portfolio_data)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Portfolio Avg ESG", f"{df['ESG Score'].mean():.1f}")
        with col2:
            st.metric("Companies Analyzed", len(df))
        with col3:
            leaders = len(df[df["ESG Score"] >= 70])
            st.metric("ESG Leaders", f"{leaders}")
        with col4:
            laggards = len(df[df["ESG Score"] < 50])
            st.metric("ESG Laggards", f"{laggards}")

        # Portfolio table
        st.markdown("### Portfolio Holdings")
        st.dataframe(
            df.style.background_gradient(subset=["ESG Score"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )

        # Sector breakdown
        st.markdown("### Sector Distribution")

        col1, col2 = st.columns(2)

        with col1:
            sector_scores = (
                df.groupby("Sector")["ESG Score"].mean().sort_values(ascending=False)
            )

            fig = px.bar(
                x=sector_scores.values,
                y=sector_scores.index,
                orientation="h",
                labels={"x": "Average ESG Score", "y": "Sector"},
                title="Average ESG Score by Sector",
                color=sector_scores.values,
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            rating_dist = df["Rating"].value_counts().sort_index()

            fig = px.pie(
                values=rating_dist.values,
                names=rating_dist.index,
                title="ESG Rating Distribution",
                color_discrete_sequence=px.colors.sequential.RdYlGn,
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
