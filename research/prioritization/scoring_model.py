from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Dict
import json


"""
Hypothesis Prioritization Scoring Model

Scores each trading hypothesis on 5 dimensions to determine testing priority.
Score = (Soundness × Measurability × Competition × Persistence) / 10,000
Adjust with Capital Capacity multiplier for final priority score.

Target: Score > 500 for testing priority
"""



class Category(Enum):
    ON_CHAIN = "On-Chain"
    MICROSTRUCTURE = "Microstructure"
    ARBITRAGE = "Arbitrage"
    SENTIMENT = "Sentiment"
    REGIME_DEPENDENT = "Regime-Dependent"


class Status(Enum):
    RESEARCH = "Research"
    TESTING = "Testing"
    VALIDATED = "Validated"
    KILLED = "Killed"
    PRODUCTION = "Production"


@dataclass
class HypothesisScore:
    """Individual scoring dimensions for a hypothesis"""
    theoretical_soundness: int  # 1-10: Is there logical reason this inefficiency exists?
    measurability: int  # 1-10: Can we get required data?
    competition: int  # 1-10: How uncrowded? (10 = no one doing this, 1 = everyone)
    persistence: int  # 1-10: Will this last or get arbitraged away?
    capital_capacity: int  # 1-10: Can we scale this strategy?
    
    def __post_init__(self):
        """Validate scores are in range"""
        for field in ['theoretical_soundness', 'measurability', 'competition', 
                      'persistence', 'capital_capacity']:
            value = getattr(self, field)
            if not 1 <= value <= 10:
                raise ValueError(f"{field} must be between 1-10, got {value}")
    
    def base_score(self) -> float:
        """Calculate base priority score (before capital capacity adjustment)"""
        return (
            self.theoretical_soundness * 
            self.measurability * 
            self.competition * 
            self.persistence
        ) / 10000
    
    def final_score(self) -> float:
        """Calculate final priority score (with capital capacity multiplier)"""
        base = self.base_score()
        # Capital capacity acts as multiplier (8/10 = 0.8x, 10/10 = 1.0x)
        capacity_multiplier = self.capital_capacity / 10.0
        return base * 1000 * capacity_multiplier
    
    def priority_tier(self) -> str:
        """Classify hypothesis into priority tier"""
        score = self.final_score()
        if score >= 700:
            return "CRITICAL"
        elif score >= 500:
            return "HIGH"
        elif score >= 300:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class Hypothesis:
    """Complete hypothesis metadata"""
    id: str  # e.g., "001"
    title: str
    category: Category
    status: Status
    score: HypothesisScore
    
    # Data requirements
    data_cost_per_month: float  # USD
    engineering_hours: int
    
    # Testing results (populated after testing)
    backtest_sharpe: float = 0.0
    backtest_win_rate: float = 0.0
    test_date: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category.value,
            "status": self.status.value,
            "score": {
                "theoretical_soundness": self.score.theoretical_soundness,
                "measurability": self.score.measurability,
                "competition": self.score.competition,
                "persistence": self.score.persistence,
                "capital_capacity": self.score.capital_capacity,
                "base_score": self.score.base_score(),
                "final_score": self.score.final_score(),
                "priority_tier": self.score.priority_tier()
            },
            "data_cost_per_month": self.data_cost_per_month,
            "engineering_hours": self.engineering_hours,
            "backtest_sharpe": self.backtest_sharpe,
            "backtest_win_rate": self.backtest_win_rate,
            "test_date": self.test_date
        }


class HypothesisDatabase:
    """Manage collection of hypotheses"""
    
    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
    
    def add(self, hypothesis: Hypothesis):
        """Add hypothesis to database"""
        self.hypotheses.append(hypothesis)
    
    @lru_cache(maxsize=128)
    
    def get_by_id(self, hypothesis_id: str) -> Hypothesis:
        """Retrieve hypothesis by ID"""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        raise ValueError(f"Hypothesis {hypothesis_id} not found")
    
    @lru_cache(maxsize=128)
    
    def get_by_priority(self, min_score: float = 0) -> List[Hypothesis]:
        """Get hypotheses sorted by priority score"""
        filtered = [h for h in self.hypotheses if h.score.final_score() >= min_score]
        return sorted(filtered, key=lambda h: h.score.final_score(), reverse=True)
    
    @lru_cache(maxsize=128)
    
    def get_by_category(self, category: Category) -> List[Hypothesis]:
        """Get all hypotheses in a category"""
        return [h for h in self.hypotheses if h.category == category]
    
    @lru_cache(maxsize=128)
    
    def get_by_status(self, status: Status) -> List[Hypothesis]:
        """Get all hypotheses with given status"""
        return [h for h in self.hypotheses if h.status == status]
    
    @lru_cache(maxsize=128)
    
    def get_top_n(self, n: int = 5) -> List[Hypothesis]:
        """Get top N hypotheses by score"""
        return self.get_by_priority()[:n]
    
    def summary_stats(self) -> Dict:
        """Calculate summary statistics"""
        if not self.hypotheses:
            return {"total": 0}
        
        scores = [h.score.final_score() for h in self.hypotheses]
        
        return {
            "total": len(self.hypotheses),
            "by_category": {
                cat.value: len(self.get_by_category(cat))
                for cat in Category
            },
            "by_status": {
                stat.value: len(self.get_by_status(stat))
                for stat in Status
            },
            "by_priority_tier": {
                "CRITICAL": len([h for h in self.hypotheses if h.score.priority_tier() == "CRITICAL"]),
                "HIGH": len([h for h in self.hypotheses if h.score.priority_tier() == "HIGH"]),
                "MEDIUM": len([h for h in self.hypotheses if h.score.priority_tier() == "MEDIUM"]),
                "LOW": len([h for h in self.hypotheses if h.score.priority_tier() == "LOW"]),
            },
            "score_stats": {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            },
            "total_data_cost": sum(h.data_cost_per_month for h in self.hypotheses),
            "total_engineering_hours": sum(h.engineering_hours for h in self.hypotheses)
        }
    
    def save_to_json(self, filepath: Path):
        """Save database to JSON file"""
        data = {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "summary": self.summary_stats()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_report(self):
        """Print priority report to console"""
        print("\n" + "="*80)
        print("HYPOTHESIS PRIORITY REPORT")
        print("="*80 + "\n")
        
        stats = self.summary_stats()
        print(f"Total Hypotheses: {stats['total']}")
        print(f"Total Data Cost: ${stats['total_data_cost']:.2f}/month")
        print(f"Total Engineering Hours: {stats['total_engineering_hours']} hours")
        print()
        
        print("By Priority Tier:")
        for tier, count in stats['by_priority_tier'].items():
            print(f"  {tier}: {count}")
        print()
        
        print("By Category:")
        for cat, count in stats['by_category'].items():
            print(f"  {cat}: {count}")
        print()
        
        print("Top 10 Hypotheses by Score:")
        print("-" * 80)
        print(f"{'ID':<5} {'Title':<40} {'Score':<8} {'Tier':<10} {'Cost':<8}")
        print("-" * 80)
        
        for h in self.get_top_n(10):
            print(f"{h.id:<5} {h.title[:38]:<40} {h.score.final_score():<8.0f} "
                  f"{h.score.priority_tier():<10} ${h.data_cost_per_month:<7.0f}")
        print("-" * 80)
        print()


# Example usage and initial hypotheses
def initialize_database() -> HypothesisDatabase:
    """Initialize database with first 3 hypotheses"""
    db = HypothesisDatabase()
    
    # Hypothesis 001: Whale Transaction Impact
    db.add(Hypothesis(
        id="001",
        title="Whale Exchange Deposits Predict Price Drops",
        category=Category.ON_CHAIN,
        status=Status.RESEARCH,
        score=HypothesisScore(
            theoretical_soundness=8,
            measurability=10,
            competition=6,
            persistence=7,
            capital_capacity=8
        ),
        data_cost_per_month=0,  # Free (Etherscan, Blockchain.com)
        engineering_hours=16
    ))
    
    # Hypothesis 002: Order Book Imbalance
    db.add(Hypothesis(
        id="002",
        title="Order Book Imbalance Predicts Short-Term Returns",
        category=Category.MICROSTRUCTURE,
        status=Status.RESEARCH,
        score=HypothesisScore(
            theoretical_soundness=9,
            measurability=9,
            competition=5,
            persistence=8,
            capital_capacity=6
        ),
        data_cost_per_month=0,  # Free (Binance, Coinbase WebSocket)
        engineering_hours=24
    ))
    
    # Hypothesis 003: CEX-DEX Arbitrage
    db.add(Hypothesis(
        id="003",
        title="CEX-DEX Price Dislocations Create Arbitrage Opportunities",
        category=Category.ARBITRAGE,
        status=Status.RESEARCH,
        score=HypothesisScore(
            theoretical_soundness=10,
            measurability=9,
            competition=6,
            persistence=9,
            capital_capacity=9
        ),
        data_cost_per_month=0,  # Free (CEX APIs + DEX subgraph)
        engineering_hours=20
    ))
    
    return db


if __name__ == "__main__":
    # Initialize with first 3 hypotheses
    db = initialize_database()
    
    # Print report
    db.print_report()
    
    # Save to JSON
    output_path = Path(__file__).parent.parent / "hypotheses" / "priority_scores.json"
    db.save_to_json(output_path)
    print(f"\n✅ Saved priority scores to: {output_path}")

