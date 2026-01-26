"""
Bayesian Probability Updater

Principle: Even a simple Bayesian framework outperforms vibes trading.

This module implements:
1. Prior probability from market prices
2. Likelihood updates from new evidence
3. Posterior probability (our trading signal)

Example:
    - Market says 60% chance CPI > 2.7%
    - Cleveland Fed nowcast comes in at 2.8%
    - We update: P(CPI > 2.7% | nowcast = 2.8%) = ?
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

from utils.logger import get_logger

logger = get_logger("bayesian")


@dataclass
class Evidence:
    """A piece of evidence that updates our probability."""
    name: str               # "cleveland_nowcast", "adp_report", etc.
    value: float            # The observed value
    expected: float         # What was expected (consensus)
    timestamp: datetime
    reliability: float      # How much to weight this evidence (0-1)

    @property
    def surprise(self) -> float:
        """How surprising is this evidence?"""
        return self.value - self.expected

    def __repr__(self):
        direction = "↑" if self.surprise > 0 else "↓" if self.surprise < 0 else "→"
        return f"{self.name}: {self.value} (expected {self.expected}) {direction}"


@dataclass
class BayesianBelief:
    """Our current belief state for a market."""
    market: str             # Market ticker
    prior: float            # Prior probability (usually from market price)
    posterior: float        # Updated probability after evidence
    evidence_used: List[Evidence]
    updated_at: datetime
    confidence: float       # How confident in our posterior (0-1)

    @property
    def edge(self) -> float:
        """Edge vs market (posterior - prior)."""
        return self.posterior - self.prior

    def __repr__(self):
        return (f"Belief({self.market}): prior={self.prior:.1%}, "
                f"posterior={self.posterior:.1%}, edge={self.edge:+.1%}")


class BayesianUpdater:
    """
    Updates probability beliefs based on evidence.

    The key insight: We don't need to predict perfectly.
    We just need to update more rigorously than the crowd.

    Usage:
        updater = BayesianUpdater()

        # Start with market prior
        prior = 0.60  # Market says 60% chance

        # Add evidence
        evidence = Evidence(
            name="cleveland_nowcast",
            value=2.78,         # Nowcast says 2.78%
            expected=2.70,      # Threshold is 2.7%
            timestamp=datetime.now(),
            reliability=0.8     # Cleveland Fed is pretty reliable
        )

        # Get updated belief
        belief = updater.update(market="CPI-ABOVE-2.7", prior=prior, evidence=[evidence])
        print(belief.posterior)  # Our updated probability
    """

    # Sensitivity parameters (should be calibrated with historical data)
    # These control how much evidence moves our probability

    # For CPI: how much does 0.1% nowcast difference change probability?
    CPI_SENSITIVITY = 0.15  # 0.1% CPI difference = 15% probability shift

    # For NFP: how much does 50K jobs difference change probability?
    NFP_SENSITIVITY = 0.10  # 50K jobs difference = 10% probability shift

    def __init__(self):
        self.beliefs: Dict[str, BayesianBelief] = {}

    def update(
        self,
        market: str,
        prior: float,
        evidence: List[Evidence],
        threshold: float = None
    ) -> BayesianBelief:
        """
        Update probability belief based on new evidence.

        Uses log-odds form of Bayes' rule for numerical stability:
        log_odds(posterior) = log_odds(prior) + sum(log_likelihood_ratios)

        Args:
            market: Market identifier
            prior: Prior probability (usually from market price)
            evidence: List of Evidence objects
            threshold: Market threshold (e.g., 2.7 for "CPI > 2.7%")

        Returns:
            BayesianBelief with updated posterior
        """
        if not evidence:
            return BayesianBelief(
                market=market,
                prior=prior,
                posterior=prior,
                evidence_used=[],
                updated_at=datetime.utcnow(),
                confidence=0.5
            )

        # Convert prior to log-odds
        prior_log_odds = self._prob_to_log_odds(prior)

        # Accumulate evidence
        total_log_likelihood_ratio = 0.0
        total_reliability = 0.0

        for ev in evidence:
            # Calculate likelihood ratio for this evidence
            llr = self._calculate_log_likelihood_ratio(ev, threshold)

            # Weight by reliability
            weighted_llr = llr * ev.reliability
            total_log_likelihood_ratio += weighted_llr
            total_reliability += ev.reliability

        # Calculate posterior
        posterior_log_odds = prior_log_odds + total_log_likelihood_ratio
        posterior = self._log_odds_to_prob(posterior_log_odds)

        # Clamp to reasonable range
        posterior = max(0.02, min(0.98, posterior))

        # Calculate confidence based on evidence quality
        avg_reliability = total_reliability / len(evidence) if evidence else 0.5
        confidence = min(0.95, avg_reliability * (1 + len(evidence) * 0.1))

        belief = BayesianBelief(
            market=market,
            prior=prior,
            posterior=posterior,
            evidence_used=evidence,
            updated_at=datetime.utcnow(),
            confidence=confidence
        )

        # Store for later reference
        self.beliefs[market] = belief

        logger.info(f"Updated belief: {belief}")

        return belief

    def _calculate_log_likelihood_ratio(
        self,
        evidence: Evidence,
        threshold: float = None
    ) -> float:
        """
        Calculate log likelihood ratio for a piece of evidence.

        LLR = log(P(evidence | hypothesis) / P(evidence | not hypothesis))

        For economic data, we use a simplified model based on
        how far the evidence is from the threshold.
        """
        surprise = evidence.surprise

        # Determine sensitivity based on evidence type
        if 'cpi' in evidence.name.lower() or 'pce' in evidence.name.lower():
            # For inflation data
            # Surprise is in percentage points (e.g., 0.1 = 0.1% higher than expected)
            sensitivity = self.CPI_SENSITIVITY

            if threshold is not None:
                # Distance from threshold matters
                distance_from_threshold = evidence.value - threshold
                # If nowcast is above threshold, YES is more likely
                llr = distance_from_threshold * sensitivity * 10
            else:
                # Just use surprise magnitude
                llr = surprise * sensitivity * 10

        elif 'nfp' in evidence.name.lower() or 'jobs' in evidence.name.lower() or 'adp' in evidence.name.lower():
            # For jobs data
            sensitivity = self.NFP_SENSITIVITY
            # Surprise is in thousands of jobs
            llr = (surprise / 50.0) * sensitivity * 5  # Normalize to 50K units

        else:
            # Generic evidence
            llr = surprise * 0.1  # Conservative default

        # Clamp to prevent extreme updates
        return max(-2.0, min(2.0, llr))

    def _prob_to_log_odds(self, p: float) -> float:
        """Convert probability to log-odds."""
        p = max(0.001, min(0.999, p))  # Avoid log(0)
        return math.log(p / (1 - p))

    def _log_odds_to_prob(self, lo: float) -> float:
        """Convert log-odds to probability."""
        return 1 / (1 + math.exp(-lo))

    def get_trading_signal(
        self,
        belief: BayesianBelief,
        min_edge: float = 0.05,
        min_confidence: float = 0.5
    ) -> Optional[Dict]:
        """
        Generate trading signal from belief.

        Args:
            belief: BayesianBelief object
            min_edge: Minimum edge to generate signal
            min_confidence: Minimum confidence to generate signal

        Returns:
            Dict with signal details, or None if no trade
        """
        if abs(belief.edge) < min_edge:
            logger.debug(f"Edge {belief.edge:.1%} below threshold {min_edge:.1%}")
            return None

        if belief.confidence < min_confidence:
            logger.debug(f"Confidence {belief.confidence:.1%} below threshold {min_confidence:.1%}")
            return None

        direction = "BUY_YES" if belief.edge > 0 else "BUY_NO"

        return {
            'market': belief.market,
            'direction': direction,
            'prior': belief.prior,
            'posterior': belief.posterior,
            'edge': belief.edge,
            'confidence': belief.confidence,
            'evidence_summary': [str(e) for e in belief.evidence_used],
            'recommendation': f"{direction} with {abs(belief.edge):.1%} edge"
        }


class PostMortem:
    """
    Track and analyze trade outcomes.

    Principle: This is how you compound an edge.

    For every trade, record:
    - Entry price and thesis
    - What evidence you used
    - What would falsify your thesis
    - Outcome

    Then analyze: Was your edge real? Were your updates correct?
    """

    def __init__(self, storage_path: str = "./data/postmortems.json"):
        self.storage_path = storage_path
        self.trades: List[Dict] = []
        self._load()

    def record_trade(
        self,
        market: str,
        direction: str,
        entry_price: float,
        thesis: str,
        evidence: List[str],
        falsification: str,
        prior: float,
        posterior: float
    ) -> str:
        """Record a trade for later analysis."""
        trade_id = f"{market}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        trade = {
            'id': trade_id,
            'market': market,
            'direction': direction,
            'entry_price': entry_price,
            'thesis': thesis,
            'evidence': evidence,
            'falsification': falsification,
            'prior': prior,
            'posterior': posterior,
            'entered_at': datetime.utcnow().isoformat(),
            'exit_price': None,
            'outcome': None,
            'actual_result': None,
            'pnl': None,
            'lessons': None
        }

        self.trades.append(trade)
        self._save()

        logger.info(f"Recorded trade: {trade_id}")
        return trade_id

    def record_outcome(
        self,
        trade_id: str,
        exit_price: float,
        actual_result: str,
        pnl: float,
        lessons: str
    ):
        """Record the outcome of a trade."""
        for trade in self.trades:
            if trade['id'] == trade_id:
                trade['exit_price'] = exit_price
                trade['actual_result'] = actual_result
                trade['pnl'] = pnl
                trade['outcome'] = 'WIN' if pnl > 0 else 'LOSS'
                trade['lessons'] = lessons
                trade['closed_at'] = datetime.utcnow().isoformat()
                self._save()
                logger.info(f"Recorded outcome for {trade_id}: {trade['outcome']}")
                return

        logger.warning(f"Trade not found: {trade_id}")

    def analyze_performance(self) -> Dict:
        """Analyze overall trading performance."""
        closed_trades = [t for t in self.trades if t['outcome'] is not None]

        if not closed_trades:
            return {'message': 'No closed trades to analyze'}

        wins = [t for t in closed_trades if t['outcome'] == 'WIN']
        losses = [t for t in closed_trades if t['outcome'] == 'LOSS']

        total_pnl = sum(t['pnl'] for t in closed_trades)
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0

        # Analyze edge accuracy
        # Did high-edge trades perform better than low-edge?
        high_edge = [t for t in closed_trades if abs(t['posterior'] - t['prior']) > 0.10]
        low_edge = [t for t in closed_trades if abs(t['posterior'] - t['prior']) <= 0.10]

        high_edge_winrate = len([t for t in high_edge if t['outcome'] == 'WIN']) / len(high_edge) if high_edge else 0
        low_edge_winrate = len([t for t in low_edge if t['outcome'] == 'WIN']) / len(low_edge) if low_edge else 0

        return {
            'total_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'high_edge_trades': len(high_edge),
            'high_edge_winrate': high_edge_winrate,
            'low_edge_trades': len(low_edge),
            'low_edge_winrate': low_edge_winrate,
            'edge_differential': high_edge_winrate - low_edge_winrate,
            'common_lessons': self._extract_common_lessons()
        }

    def _extract_common_lessons(self) -> List[str]:
        """Extract common patterns from lessons learned."""
        lessons = [t.get('lessons', '') for t in self.trades if t.get('lessons')]
        # In a real implementation, could use NLP to cluster similar lessons
        return lessons[-5:] if lessons else []  # Return last 5 lessons

    def _load(self):
        """Load trades from storage."""
        import json
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.trades = json.load(f)
            except:
                self.trades = []

    def _save(self):
        """Save trades to storage."""
        import json
        import os
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.trades, f, indent=2)


# Import os for PostMortem
import os


if __name__ == "__main__":
    print("Testing Bayesian Updater...")

    updater = BayesianUpdater()

    # Scenario: CPI market at 60% for "above 2.7%"
    # Cleveland Fed nowcast comes in at 2.78%
    print("\nScenario 1: CPI nowcast above threshold")
    evidence = [
        Evidence(
            name="cleveland_nowcast",
            value=2.78,
            expected=2.70,
            timestamp=datetime.now(),
            reliability=0.8
        )
    ]

    belief = updater.update(
        market="CPI-ABOVE-2.7",
        prior=0.60,
        evidence=evidence,
        threshold=2.7
    )

    print(f"  Prior (market): 60%")
    print(f"  Evidence: Cleveland nowcast = 2.78% (threshold = 2.7%)")
    print(f"  Posterior: {belief.posterior:.1%}")
    print(f"  Edge: {belief.edge:+.1%}")

    signal = updater.get_trading_signal(belief)
    if signal:
        print(f"  Signal: {signal['recommendation']}")

    # Scenario 2: ADP report surprises to upside
    print("\nScenario 2: ADP jobs beat")
    evidence = [
        Evidence(
            name="adp_report",
            value=150,        # 150K jobs
            expected=75,      # Expected 75K
            timestamp=datetime.now(),
            reliability=0.6   # ADP is less reliable than NFP
        )
    ]

    belief = updater.update(
        market="NFP-ABOVE-100K",
        prior=0.50,
        evidence=evidence
    )

    print(f"  Prior (market): 50%")
    print(f"  Evidence: ADP = +150K (expected +75K)")
    print(f"  Posterior: {belief.posterior:.1%}")
    print(f"  Edge: {belief.edge:+.1%}")

    # Test post-mortem
    print("\nTesting PostMortem tracking...")
    pm = PostMortem("./data/test_postmortems.json")

    trade_id = pm.record_trade(
        market="CPI-ABOVE-2.7",
        direction="BUY_YES",
        entry_price=0.60,
        thesis="Cleveland nowcast at 2.78% suggests higher probability",
        evidence=["Cleveland Fed nowcast: 2.78%"],
        falsification="Would exit if revised nowcast drops below 2.65%",
        prior=0.60,
        posterior=0.72
    )

    # Simulate outcome
    pm.record_outcome(
        trade_id=trade_id,
        exit_price=0.75,
        actual_result="CPI came in at 2.8%",
        pnl=0.15,
        lessons="Cleveland Fed nowcast was accurate; trust it more in future"
    )

    print(f"\nPerformance analysis:")
    analysis = pm.analyze_performance()
    for k, v in analysis.items():
        print(f"  {k}: {v}")
