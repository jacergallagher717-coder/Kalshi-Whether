#!/usr/bin/env python3
"""
Diagnostic script to understand why signals are being filtered.
Run this on the server to see exactly what's blocking each market.
"""

from datetime import date, timedelta
from collections import defaultdict
from config.locations import ACTIVE_LOCATIONS
from config.settings import (
    MIN_EDGE_THRESHOLD, MIN_YES_PRICE, MAX_YES_PRICE, MAX_NO_PRICE_BRACKET,
    MAX_MODEL_SPREAD, MIN_CONFIDENCE_SCORE, MIN_MODEL_AGREEMENT, MIN_MODELS_REQUIRED,
    MIN_PROBABILITY_THRESHOLD, MAX_FORECAST_DAYS
)
from data.collectors.data_manager import DataManager
from models.probability import ensemble_probability
from utils.helpers import calculate_days_until

def diagnose():
    print("=" * 70)
    print("SIGNAL DIAGNOSTIC - Why aren't more signals being generated?")
    print("=" * 70)

    # Show current filter settings
    print("\n📋 CURRENT FILTER SETTINGS:")
    print(f"  MIN_EDGE_THRESHOLD: {MIN_EDGE_THRESHOLD:.0%}")
    print(f"  MIN_PROBABILITY_THRESHOLD: {MIN_PROBABILITY_THRESHOLD:.0%}")
    print(f"  MIN_CONFIDENCE_SCORE: {MIN_CONFIDENCE_SCORE:.0%}")
    print(f"  MIN_MODEL_AGREEMENT: {MIN_MODEL_AGREEMENT:.0%}")
    print(f"  MIN_MODELS_REQUIRED: {MIN_MODELS_REQUIRED}")
    print(f"  MAX_MODEL_SPREAD: {MAX_MODEL_SPREAD}°F")
    print(f"  MAX_FORECAST_DAYS: {MAX_FORECAST_DAYS}")
    print(f"  Price range: ${MIN_YES_PRICE:.2f} - ${MAX_YES_PRICE:.2f}")

    dm = DataManager()

    # Stats tracking
    total_markets = 0
    filter_stats = defaultdict(int)
    potential_signals = []

    for loc in ACTIVE_LOCATIONS:
        print(f"\n{'='*60}")
        print(f"📍 {loc}")
        print("=" * 60)

        try:
            markets = dm.get_markets(loc)
            if not markets:
                print(f"  ❌ No markets found")
                continue

            print(f"  Found {len(markets)} markets")
            total_markets += len(markets)

            # Group by date
            today = date.today()
            markets_by_date = defaultdict(list)
            for m in markets:
                markets_by_date[m.target_date].append(m)

            for target_date in sorted(markets_by_date.keys()):
                days_out = (target_date - today).days
                date_markets = markets_by_date[target_date]

                # Skip expired
                if days_out < 0:
                    filter_stats['expired'] += len(date_markets)
                    continue

                # Skip too far out
                if days_out > MAX_FORECAST_DAYS:
                    filter_stats['too_far_out'] += len(date_markets)
                    continue

                print(f"\n  📅 {target_date} ({days_out} days out) - {len(date_markets)} markets")

                # Get forecasts for this date
                try:
                    forecasts = dm.weather_client.get_forecasts_for_date(loc, target_date)
                except Exception as e:
                    print(f"    ❌ Forecast error: {e}")
                    filter_stats['no_forecasts'] += len(date_markets)
                    continue

                if not forecasts:
                    print(f"    ❌ No forecasts available")
                    filter_stats['no_forecasts'] += len(date_markets)
                    continue

                # Show forecasts
                print(f"    Forecasts ({len(forecasts)} sources):")
                for src, f in forecasts.items():
                    print(f"      {src}: High={f.high_temp_f}°F, Low={f.low_temp_f}°F")

                # Check each market
                for m in date_markets:
                    # Get relevant temp forecasts
                    temp_forecasts = {}
                    for src, f in forecasts.items():
                        temp = f.high_temp_f if m.market_type == 'high' else f.low_temp_f
                        if temp is not None:
                            temp_forecasts[src] = temp

                    # Calculate what we need
                    if len(temp_forecasts) < MIN_MODELS_REQUIRED:
                        filter_stats['not_enough_models'] += 1
                        continue

                    temps = list(temp_forecasts.values())
                    spread = max(temps) - min(temps)
                    avg_temp = sum(temps) / len(temps)

                    # Check model spread
                    if spread > MAX_MODEL_SPREAD:
                        filter_stats['model_spread'] += 1
                        continue

                    # Model agreement score
                    agreement_score = max(0.0, 1.0 - spread / 10.0)
                    if agreement_score < MIN_MODEL_AGREEMENT:
                        filter_stats['model_agreement'] += 1
                        continue

                    # Calculate probability
                    direction = 'above' if m.market_type == 'high' else 'below'
                    is_bracket = '-B' in m.ticker
                    result = ensemble_probability(
                        temp_forecasts, m.temp_threshold, days_out, direction,
                        city=loc, is_bracket=is_bracket, market_type=m.market_type
                    )
                    our_prob = result['ensemble_prob']
                    confidence = result['confidence']

                    # Calculate edge and direction
                    raw_edge = our_prob - m.yes_price
                    if raw_edge > 0:
                        trade_dir = "BUY_YES"
                        edge = raw_edge
                        bet_prob = our_prob
                        entry_price = m.yes_price
                    else:
                        trade_dir = "BUY_NO"
                        edge = abs(raw_edge)
                        bet_prob = 1 - our_prob
                        entry_price = 1 - m.yes_price

                    # Now check all filters
                    reasons = []

                    # Edge threshold (adjusted for same-day)
                    edge_threshold = MIN_EDGE_THRESHOLD
                    if days_out == 0:
                        edge_threshold = max(edge_threshold, 0.20)

                    if edge < edge_threshold:
                        reasons.append(f"edge {edge:.1%} < {edge_threshold:.0%}")
                        filter_stats['edge_too_low'] += 1

                    # Probability filter
                    if bet_prob < MIN_PROBABILITY_THRESHOLD:
                        reasons.append(f"bet_prob {bet_prob:.1%} < {MIN_PROBABILITY_THRESHOLD:.0%}")
                        filter_stats['probability_filter'] += 1

                    # Confidence filter
                    if confidence < MIN_CONFIDENCE_SCORE:
                        reasons.append(f"confidence {confidence:.1%} < {MIN_CONFIDENCE_SCORE:.0%}")
                        filter_stats['confidence_filter'] += 1

                    # Sweet spot filter
                    if trade_dir == "BUY_YES":
                        if m.yes_price < MIN_YES_PRICE:
                            reasons.append(f"YES ${m.yes_price:.2f} < min ${MIN_YES_PRICE:.2f}")
                            filter_stats['price_too_low'] += 1
                        if m.yes_price > MAX_YES_PRICE:
                            reasons.append(f"YES ${m.yes_price:.2f} > max ${MAX_YES_PRICE:.2f}")
                            filter_stats['price_too_high'] += 1
                    else:  # BUY_NO
                        no_price = 1 - m.yes_price
                        if no_price < MIN_YES_PRICE:
                            reasons.append(f"NO ${no_price:.2f} < min ${MIN_YES_PRICE:.2f}")
                            filter_stats['price_too_low'] += 1
                        if no_price > MAX_YES_PRICE:
                            reasons.append(f"NO ${no_price:.2f} > max ${MAX_YES_PRICE:.2f}")
                            filter_stats['price_too_high'] += 1
                        if is_bracket and no_price > MAX_NO_PRICE_BRACKET:
                            reasons.append(f"bracket NO ${no_price:.2f} > max ${MAX_NO_PRICE_BRACKET:.2f}")
                            filter_stats['bracket_price'] += 1

                    # Print market analysis
                    ticker_short = m.ticker.split('-')[0] + '-' + m.ticker.split('-')[1] + '-' + m.ticker.split('-')[2][:4]
                    if reasons:
                        # Show promising but filtered markets (edge > 10%)
                        if edge > 0.10:
                            print(f"    ⚠️  {ticker_short}: {trade_dir} @ ${entry_price:.2f}")
                            print(f"        Edge: {edge:.1%}, Forecast: {avg_temp:.0f}°F vs threshold {m.temp_threshold}°F")
                            print(f"        FILTERED: {', '.join(reasons)}")
                    else:
                        # SIGNAL!
                        print(f"    ✅ {m.ticker}")
                        print(f"        {trade_dir} @ ${entry_price:.2f}, Edge: {edge:.1%}")
                        print(f"        Forecast: {avg_temp:.0f}°F, Threshold: {m.temp_threshold}°F")
                        potential_signals.append({
                            'ticker': m.ticker,
                            'direction': trade_dir,
                            'price': entry_price,
                            'edge': edge,
                            'forecast': avg_temp,
                            'threshold': m.temp_threshold
                        })

        except Exception as e:
            import traceback
            print(f"  ❌ Error: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\nTotal markets scanned: {total_markets}")
    print(f"Signals found: {len(potential_signals)}")

    print("\n📉 FILTER BREAKDOWN:")
    for reason, count in sorted(filter_stats.items(), key=lambda x: -x[1]):
        pct = count / total_markets * 100 if total_markets > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    if potential_signals:
        print("\n✅ VALID SIGNALS:")
        for sig in potential_signals:
            print(f"  {sig['ticker']}: {sig['direction']} @ ${sig['price']:.2f}, Edge: {sig['edge']:.1%}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if filter_stats.get('edge_too_low', 0) > total_markets * 0.5:
        print("  - Most markets filtered by EDGE. Market prices match forecasts closely.")
        print("    This is normal - it means markets are efficient. Wait for mispricings.")
    if filter_stats.get('price_too_high', 0) > 10:
        print(f"  - {filter_stats.get('price_too_high', 0)} filtered by HIGH PRICE.")
        print(f"    Consider increasing MAX_YES_PRICE above {MAX_YES_PRICE:.0%}")
    if filter_stats.get('confidence_filter', 0) > 10:
        print(f"  - {filter_stats.get('confidence_filter', 0)} filtered by CONFIDENCE.")
        print(f"    Consider lowering MIN_CONFIDENCE_SCORE below {MIN_CONFIDENCE_SCORE:.0%}")

if __name__ == "__main__":
    diagnose()
