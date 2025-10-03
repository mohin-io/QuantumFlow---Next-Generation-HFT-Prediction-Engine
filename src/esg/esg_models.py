"""
ESG (Environmental, Social, Governance) Data Models and Scoring System

Implements comprehensive ESG metrics and scoring methodology.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ESGMetrics:
    """ESG metrics for a company."""

    # Environmental
    co2_emissions: float  # tons per year
    water_usage: float  # cubic meters
    renewable_energy_pct: float  # percentage
    waste_recycled_pct: float  # percentage
    environmental_violations: int

    # Social
    employee_satisfaction: float  # 0-100 scale
    diversity_score: float  # 0-100 scale
    safety_incidents: int
    community_investment: float  # $ millions
    supply_chain_ethics_score: float  # 0-100

    # Governance
    board_independence: float  # percentage
    female_board_members: float  # percentage
    executive_compensation_ratio: float  # CEO to median employee
    audit_quality_score: float  # 0-100
    shareholder_rights_score: float  # 0-100

    # Financial
    revenue: float  # $ millions
    market_cap: float  # $ millions

    # Metadata
    company: str
    sector: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ESGScorer:
    """Calculate ESG scores using weighted methodology."""

    def __init__(self):
        # Define weights for each pillar
        self.weights = {
            'environmental': 0.35,
            'social': 0.35,
            'governance': 0.30
        }

        # Benchmark values (industry averages)
        self.benchmarks = {
            'co2_intensity': 500,  # tons per $M revenue
            'water_intensity': 1000,  # cubic meters per $M revenue
            'renewable_energy_pct': 30,
            'waste_recycled_pct': 50,
            'employee_satisfaction': 70,
            'diversity_score': 60,
            'board_independence': 75,
            'female_board_members': 30,
            'executive_compensation_ratio': 150,
        }

    def calculate_environmental_score(self, metrics: ESGMetrics) -> Dict:
        """Calculate environmental pillar score (0-100)."""

        # CO2 intensity (lower is better)
        co2_intensity = metrics.co2_emissions / (metrics.revenue + 1)
        co2_score = max(0, 100 - (co2_intensity / self.benchmarks['co2_intensity']) * 100)

        # Water intensity (lower is better)
        water_intensity = metrics.water_usage / (metrics.revenue + 1)
        water_score = max(0, 100 - (water_intensity / self.benchmarks['water_intensity']) * 100)

        # Renewable energy (higher is better)
        renewable_score = min(100, (metrics.renewable_energy_pct / 100) * 100)

        # Waste recycling (higher is better)
        waste_score = min(100, (metrics.waste_recycled_pct / 100) * 100)

        # Violations penalty
        violation_penalty = min(50, metrics.environmental_violations * 10)

        # Weighted average
        env_score = (
            co2_score * 0.30 +
            water_score * 0.20 +
            renewable_score * 0.25 +
            waste_score * 0.20 -
            violation_penalty * 0.05
        )

        return {
            'score': max(0, min(100, env_score)),
            'components': {
                'co2_score': co2_score,
                'water_score': water_score,
                'renewable_score': renewable_score,
                'waste_score': waste_score,
                'violation_penalty': violation_penalty
            }
        }

    def calculate_social_score(self, metrics: ESGMetrics) -> Dict:
        """Calculate social pillar score (0-100)."""

        # Employee satisfaction (0-100 scale)
        satisfaction_score = metrics.employee_satisfaction

        # Diversity (0-100 scale)
        diversity_score = metrics.diversity_score

        # Safety (lower incidents is better)
        safety_score = max(0, 100 - metrics.safety_incidents * 5)

        # Community investment (higher is better, relative to revenue)
        community_ratio = (metrics.community_investment / (metrics.revenue + 1)) * 100
        community_score = min(100, community_ratio * 50)

        # Supply chain ethics
        ethics_score = metrics.supply_chain_ethics_score

        # Weighted average
        social_score = (
            satisfaction_score * 0.25 +
            diversity_score * 0.25 +
            safety_score * 0.20 +
            community_score * 0.15 +
            ethics_score * 0.15
        )

        return {
            'score': max(0, min(100, social_score)),
            'components': {
                'satisfaction_score': satisfaction_score,
                'diversity_score': diversity_score,
                'safety_score': safety_score,
                'community_score': community_score,
                'ethics_score': ethics_score
            }
        }

    def calculate_governance_score(self, metrics: ESGMetrics) -> Dict:
        """Calculate governance pillar score (0-100)."""

        # Board independence (higher is better)
        independence_score = metrics.board_independence

        # Gender diversity on board (higher is better)
        diversity_score = min(100, (metrics.female_board_members / 50) * 100)

        # Executive compensation (lower ratio is better)
        comp_ratio = metrics.executive_compensation_ratio
        benchmark_ratio = self.benchmarks['executive_compensation_ratio']
        comp_score = max(0, 100 - ((comp_ratio - benchmark_ratio) / benchmark_ratio) * 50)

        # Audit quality
        audit_score = metrics.audit_quality_score

        # Shareholder rights
        shareholder_score = metrics.shareholder_rights_score

        # Weighted average
        gov_score = (
            independence_score * 0.25 +
            diversity_score * 0.20 +
            comp_score * 0.20 +
            audit_score * 0.20 +
            shareholder_score * 0.15
        )

        return {
            'score': max(0, min(100, gov_score)),
            'components': {
                'independence_score': independence_score,
                'diversity_score': diversity_score,
                'comp_score': comp_score,
                'audit_score': audit_score,
                'shareholder_score': shareholder_score
            }
        }

    def calculate_overall_score(self, metrics: ESGMetrics) -> Dict:
        """Calculate overall ESG score with all three pillars."""

        env_result = self.calculate_environmental_score(metrics)
        social_result = self.calculate_social_score(metrics)
        gov_result = self.calculate_governance_score(metrics)

        # Overall weighted score
        overall_score = (
            env_result['score'] * self.weights['environmental'] +
            social_result['score'] * self.weights['social'] +
            gov_result['score'] * self.weights['governance']
        )

        # Rating classification
        if overall_score >= 80:
            rating = 'AAA'
            rating_desc = 'Leader'
        elif overall_score >= 70:
            rating = 'AA'
            rating_desc = 'Advanced'
        elif overall_score >= 60:
            rating = 'A'
            rating_desc = 'Good'
        elif overall_score >= 50:
            rating = 'BBB'
            rating_desc = 'Average'
        elif overall_score >= 40:
            rating = 'BB'
            rating_desc = 'Below Average'
        else:
            rating = 'B'
            rating_desc = 'Laggard'

        return {
            'overall_score': overall_score,
            'rating': rating,
            'rating_description': rating_desc,
            'pillar_scores': {
                'environmental': env_result['score'],
                'social': social_result['score'],
                'governance': gov_result['score']
            },
            'detailed_components': {
                'environmental': env_result['components'],
                'social': social_result['components'],
                'governance': gov_result['components']
            }
        }

    def get_rating_color(self, rating: str) -> str:
        """Get color code for rating."""
        colors = {
            'AAA': '#00A86B',  # Dark green
            'AA': '#50C878',   # Green
            'A': '#90EE90',    # Light green
            'BBB': '#FFD700',  # Gold
            'BB': '#FFA500',   # Orange
            'B': '#FF6347'     # Red
        }
        return colors.get(rating, '#808080')


def generate_sample_companies() -> List[ESGMetrics]:
    """Generate sample company ESG data for demonstration."""

    np.random.seed(42)

    companies = [
        {
            'company': 'TechCorp Inc.',
            'sector': 'Technology',
            'revenue': 50000,
            'market_cap': 200000,
            'co2_emissions': 15000,
            'water_usage': 30000,
            'renewable_energy_pct': 85,
            'waste_recycled_pct': 70,
            'environmental_violations': 0,
            'employee_satisfaction': 85,
            'diversity_score': 78,
            'safety_incidents': 2,
            'community_investment': 100,
            'supply_chain_ethics_score': 82,
            'board_independence': 88,
            'female_board_members': 42,
            'executive_compensation_ratio': 120,
            'audit_quality_score': 90,
            'shareholder_rights_score': 88
        },
        {
            'company': 'EnergyMax Corp.',
            'sector': 'Energy',
            'revenue': 80000,
            'market_cap': 150000,
            'co2_emissions': 120000,
            'water_usage': 500000,
            'renewable_energy_pct': 25,
            'waste_recycled_pct': 45,
            'environmental_violations': 3,
            'employee_satisfaction': 68,
            'diversity_score': 55,
            'safety_incidents': 8,
            'community_investment': 120,
            'supply_chain_ethics_score': 70,
            'board_independence': 72,
            'female_board_members': 25,
            'executive_compensation_ratio': 180,
            'audit_quality_score': 75,
            'shareholder_rights_score': 70
        },
        {
            'company': 'GreenBank Ltd.',
            'sector': 'Financial Services',
            'revenue': 30000,
            'market_cap': 100000,
            'co2_emissions': 8000,
            'water_usage': 15000,
            'renewable_energy_pct': 95,
            'waste_recycled_pct': 85,
            'environmental_violations': 0,
            'employee_satisfaction': 82,
            'diversity_score': 85,
            'safety_incidents': 1,
            'community_investment': 80,
            'supply_chain_ethics_score': 88,
            'board_independence': 90,
            'female_board_members': 48,
            'executive_compensation_ratio': 95,
            'audit_quality_score': 92,
            'shareholder_rights_score': 90
        },
        {
            'company': 'MineralCo Resources',
            'sector': 'Mining',
            'revenue': 45000,
            'market_cap': 60000,
            'co2_emissions': 180000,
            'water_usage': 800000,
            'renewable_energy_pct': 15,
            'waste_recycled_pct': 35,
            'environmental_violations': 5,
            'employee_satisfaction': 58,
            'diversity_score': 48,
            'safety_incidents': 12,
            'community_investment': 50,
            'supply_chain_ethics_score': 62,
            'board_independence': 68,
            'female_board_members': 18,
            'executive_compensation_ratio': 220,
            'audit_quality_score': 68,
            'shareholder_rights_score': 65
        },
        {
            'company': 'ConsumerGoods Global',
            'sector': 'Consumer Goods',
            'revenue': 60000,
            'market_cap': 180000,
            'co2_emissions': 35000,
            'water_usage': 120000,
            'renewable_energy_pct': 55,
            'waste_recycled_pct': 60,
            'environmental_violations': 1,
            'employee_satisfaction': 75,
            'diversity_score': 72,
            'safety_incidents': 4,
            'community_investment': 90,
            'supply_chain_ethics_score': 76,
            'board_independence': 80,
            'female_board_members': 38,
            'executive_compensation_ratio': 145,
            'audit_quality_score': 82,
            'shareholder_rights_score': 80
        }
    ]

    return [ESGMetrics(**company) for company in companies]


if __name__ == '__main__':
    print('='*80)
    print('ESG SCORING SYSTEM DEMONSTRATION')
    print('='*80)

    # Generate sample companies
    companies = generate_sample_companies()
    scorer = ESGScorer()

    print('\nCompany ESG Scores:\n')
    print(f"{'Company':<25} {'Overall':<10} {'Rating':<8} {'E':<8} {'S':<8} {'G':<8}")
    print('-'*80)

    for company in companies:
        result = scorer.calculate_overall_score(company)
        print(f"{company.company:<25} "
              f"{result['overall_score']:>6.1f}    "
              f"{result['rating']:<8} "
              f"{result['pillar_scores']['environmental']:>6.1f}  "
              f"{result['pillar_scores']['social']:>6.1f}  "
              f"{result['pillar_scores']['governance']:>6.1f}")

    print('\n' + '='*80)
