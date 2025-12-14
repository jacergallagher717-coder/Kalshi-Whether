"""
Edge validation analysis to determine if our edge is real.

Key questions:
1. Is our win rate statistically better than random?
2. Is our P&L statistically positive?
3. Do we have enough data to be confident?
"""

import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

from config.settings import TRADES_DB
from utils.logger import get_logger
from utils.helpers import format_percent

logger = get_logger("edge_validation")


@dataclass
class ValidationResult:
    """Result of statistical validation."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # p < 0.05
    interpretation: str
    sample_size: int


class EdgeValidator:
    """
    Statistical validation of trading edge.

    Performs tests to determine if observed edge is:
    - Statistically significant
    - Likely to persist
    - Sufficient for live trading
    """

    SIGNIFICANCE_LEVEL = 0.05  # 5% significance
    MIN_TRADES_FOR_VALIDATION = 20

    def __init__(self, db_path: str = None):
        """Initialize the validator."""
        self.db_path = db_path or TRADES_DB
        logger.info("Edge validator initialized")

    def _get_trade_data(self) -> Tuple[List[float], List[bool]]:
        """Get P&L and win/loss data from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT net_pnl, edge_at_entry
            FROM paper_trades
            WHERE status = 'SETTLED'
            ORDER BY settled_at ASC
        ''')

        rows = cursor.fetchall()
        conn.close()

        pnls = [row[0] for row in rows]
        wins = [pnl > 0 for pnl in pnls]

        return pnls, wins

    def test_win_rate(self) -> ValidationResult:
        """
        Test if win rate is significantly better than 50%.

        Uses binomial test: H0: p = 0.5, H1: p > 0.5
        """
        pnls, wins = self._get_trade_data()
        n = len(wins)

        if n < self.MIN_TRADES_FOR_VALIDATION:
            return ValidationResult(
                test_name="Win Rate (Binomial)",
                statistic=sum(wins) / n if n > 0 else 0,
                p_value=1.0,
                significant=False,
                interpretation=f"Insufficient data ({n} trades, need {self.MIN_TRADES_FOR_VALIDATION})",
                sample_size=n
            )

        wins_count = sum(wins)
        win_rate = wins_count / n

        # One-sided binomial test
        p_value = stats.binom_test(wins_count, n, p=0.5, alternative='greater')

        significant = p_value < self.SIGNIFICANCE_LEVEL

        if significant:
            interpretation = f"Win rate {win_rate:.1%} is significantly > 50% (p={p_value:.4f})"
        else:
            interpretation = f"Win rate {win_rate:.1%} is not significantly > 50% (p={p_value:.4f})"

        return ValidationResult(
            test_name="Win Rate (Binomial)",
            statistic=win_rate,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation,
            sample_size=n
        )

    def test_mean_pnl(self) -> ValidationResult:
        """
        Test if mean P&L is significantly greater than zero.

        Uses one-sample t-test: H0: mu = 0, H1: mu > 0
        """
        pnls, _ = self._get_trade_data()
        n = len(pnls)

        if n < self.MIN_TRADES_FOR_VALIDATION:
            return ValidationResult(
                test_name="Mean P&L (t-test)",
                statistic=np.mean(pnls) if pnls else 0,
                p_value=1.0,
                significant=False,
                interpretation=f"Insufficient data ({n} trades, need {self.MIN_TRADES_FOR_VALIDATION})",
                sample_size=n
            )

        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)

        # One-sample t-test
        t_stat, p_value_two_sided = stats.ttest_1samp(pnls, 0)
        p_value = p_value_two_sided / 2  # One-sided

        # Adjust for direction
        if mean_pnl < 0:
            p_value = 1 - p_value

        significant = p_value < self.SIGNIFICANCE_LEVEL and mean_pnl > 0

        if significant:
            interpretation = f"Mean P&L ${mean_pnl:.2f} is significantly > $0 (p={p_value:.4f})"
        else:
            interpretation = f"Mean P&L ${mean_pnl:.2f} is not significantly > $0 (p={p_value:.4f})"

        return ValidationResult(
            test_name="Mean P&L (t-test)",
            statistic=mean_pnl,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation,
            sample_size=n
        )

    def test_edge_correlation(self) -> ValidationResult:
        """
        Test if higher predicted edge correlates with better outcomes.

        Uses Pearson correlation between edge_at_entry and net_pnl.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT edge_at_entry, net_pnl
            FROM paper_trades
            WHERE status = 'SETTLED'
        ''')

        rows = cursor.fetchall()
        conn.close()

        n = len(rows)

        if n < self.MIN_TRADES_FOR_VALIDATION:
            return ValidationResult(
                test_name="Edge-PnL Correlation",
                statistic=0,
                p_value=1.0,
                significant=False,
                interpretation=f"Insufficient data ({n} trades)",
                sample_size=n
            )

        edges = [abs(row[0]) for row in rows]  # Use absolute edge
        pnls = [row[1] for row in rows]

        # Pearson correlation
        corr, p_value = stats.pearsonr(edges, pnls)

        significant = p_value < self.SIGNIFICANCE_LEVEL and corr > 0

        if significant:
            interpretation = f"Edge predicts P&L (r={corr:.3f}, p={p_value:.4f})"
        else:
            interpretation = f"Edge does not significantly predict P&L (r={corr:.3f}, p={p_value:.4f})"

        return ValidationResult(
            test_name="Edge-PnL Correlation",
            statistic=corr,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation,
            sample_size=n
        )

    def calculate_required_sample_size(
        self,
        target_win_rate: float = 0.55,
        power: float = 0.80
    ) -> int:
        """
        Calculate required sample size to detect a given win rate.

        Args:
            target_win_rate: Win rate we're trying to detect
            power: Statistical power (probability of detecting effect if it exists)

        Returns:
            Required number of trades
        """
        from scipy.stats import norm

        p0 = 0.50  # Null hypothesis (random)
        p1 = target_win_rate  # Alternative
        alpha = self.SIGNIFICANCE_LEVEL

        # Z-scores for alpha and power
        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)

        # Sample size formula for proportions
        effect_size = p1 - p0
        pooled_var = p0 * (1 - p0) + p1 * (1 - p1)

        n = ((z_alpha * np.sqrt(2 * p0 * (1 - p0)) +
              z_beta * np.sqrt(pooled_var)) / effect_size) ** 2

        return int(np.ceil(n))

    def run_all_validations(self) -> Dict[str, ValidationResult]:
        """Run all validation tests."""
        return {
            "win_rate": self.test_win_rate(),
            "mean_pnl": self.test_mean_pnl(),
            "edge_correlation": self.test_edge_correlation()
        }

    def get_validation_summary(self) -> Dict:
        """
        Get comprehensive validation summary.

        Returns recommendation on whether edge is validated.
        """
        results = self.run_all_validations()
        pnls, wins = self._get_trade_data()

        n = len(pnls)
        required_n = self.calculate_required_sample_size(0.55, 0.80)

        # Count significant tests
        significant_count = sum(1 for r in results.values() if r.significant)

        # Determine overall status
        if n < self.MIN_TRADES_FOR_VALIDATION:
            status = "INSUFFICIENT_DATA"
            recommendation = f"Need at least {self.MIN_TRADES_FOR_VALIDATION} trades to begin validation"
        elif significant_count >= 2:
            status = "VALIDATED"
            recommendation = "Edge appears statistically significant. Consider cautious live trading."
        elif significant_count == 1:
            status = "PARTIALLY_VALIDATED"
            recommendation = "Some evidence of edge. Continue paper trading for more data."
        else:
            status = "NOT_VALIDATED"
            if n < required_n:
                recommendation = f"No significant edge detected. Need ~{required_n} trades for 80% power."
            else:
                recommendation = "No significant edge detected with sufficient data. Re-evaluate strategy."

        return {
            "status": status,
            "recommendation": recommendation,
            "sample_size": n,
            "required_sample_size": required_n,
            "tests_passed": significant_count,
            "tests_total": len(results),
            "results": {name: {
                "statistic": r.statistic,
                "p_value": r.p_value,
                "significant": r.significant,
                "interpretation": r.interpretation
            } for name, r in results.items()},
            "summary_stats": {
                "win_rate": sum(wins) / n if n > 0 else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "std_pnl": np.std(pnls) if pnls else 0
            }
        }

    def generate_validation_report(self) -> str:
        """Generate text validation report."""
        summary = self.get_validation_summary()

        lines = [
            "=" * 60,
            "EDGE VALIDATION REPORT",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
            f"Status: {summary['status']}",
            f"Recommendation: {summary['recommendation']}",
            "",
            f"Sample Size: {summary['sample_size']} trades",
            f"Required for 80% Power: ~{summary['required_sample_size']} trades",
            "",
            "STATISTICAL TESTS",
            "-" * 40
        ]

        for name, result in summary["results"].items():
            status = "PASS" if result["significant"] else "FAIL"
            lines.append(f"  [{status}] {name}")
            lines.append(f"       {result['interpretation']}")
            lines.append("")

        lines.extend([
            "SUMMARY STATISTICS",
            "-" * 40,
            f"  Win Rate: {format_percent(summary['summary_stats']['win_rate'])}",
            f"  Total P&L: ${summary['summary_stats']['total_pnl']:.2f}",
            f"  Avg P&L: ${summary['summary_stats']['avg_pnl']:.2f}",
            f"  Std Dev: ${summary['summary_stats']['std_pnl']:.2f}",
            "",
            "=" * 60
        ])

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing EdgeValidator...")

    validator = EdgeValidator()

    # Run validations
    summary = validator.get_validation_summary()
    print(f"\nValidation Status: {summary['status']}")
    print(f"Recommendation: {summary['recommendation']}")

    # Generate report
    print("\n" + validator.generate_validation_report())

    # Sample size calculation
    required = validator.calculate_required_sample_size(0.55, 0.80)
    print(f"\nTo detect 55% win rate with 80% power: need ~{required} trades")
