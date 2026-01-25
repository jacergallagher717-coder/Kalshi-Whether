# Kalshi Macro Shock Trading Strategy

## Core Thesis

We profit from **timing + structure + spreads**, not headline prediction.

### The Edge Sources
1. **Nowcast Aggregation** - Combine Cleveland Fed, NY Fed, Blue Chip that retail ignores
2. **Information Calendar** - Pre-position before scheduled releases
3. **Microstructure** - Use limit orders to become the spread, not pay it
4. **Ladder Arbitrage** - Find inconsistent probabilities across adjacent strikes

---

## Target Markets

### Primary: CPI/Inflation Markets
- Monthly release: ~15th of each month, 8:30 AM ET
- Kalshi showed **50-60% lower MAE** during shock events
- Multiple nowcast sources available for aggregation

### Secondary: Fed Decision Markets
- FOMC schedule is known 1 year ahead
- Fed Funds futures provide anchor pricing
- Look for divergence between Kalshi and CME pricing

### Tertiary: Jobs/NFP Markets
- First Friday of each month, 8:30 AM ET
- ADP report provides pre-signal (2 days before)

---

## Data Sources to Aggregate

### For CPI:
1. **Cleveland Fed Inflation Nowcast** - Updates daily
   - URL: https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting
   - Provides: CPI, Core CPI estimates

2. **NY Fed Nowcast** - Weekly updates
   - URL: https://www.newyorkfed.org/research/policy/nowcast

3. **Blue Chip Consensus** - Monthly
   - Professional forecaster average

4. **CME FedWatch** - Real-time
   - Implied probabilities from Fed Funds futures

### For Fed Decisions:
1. **CME FedWatch Tool** - Definitive market pricing
2. **Fed Funds Futures** - Direct market signal
3. **Fed dot plots** - FOMC projections

---

## Execution Strategy

### Rule 1: Limit Orders Only
- Never pay the spread
- Set limit orders at fair value
- Let impatient traders come to us
- Target: capture 2-5% spread instead of paying it

### Rule 2: Pre-Position Before Information Drops
- Enter positions 24-48 hours before release
- When our nowcast aggregate diverges from market price
- Exit within minutes of release (or hold if correct)

### Rule 3: Size Based on Divergence
- Small divergence (1-2%): No trade (spread eats edge)
- Medium divergence (3-5%): Small position via limit
- Large divergence (5%+): Full position, more aggressive limits

### Rule 4: Ladder Consistency Checks
- If "CPI > 3.0%" = 40% and "CPI > 3.1%" = 45%, that's inconsistent
- Adjacent strikes should have monotonic probabilities
- Exploit mispricings between strikes

---

## Information Calendar (January 2026)

| Date | Event | Time (ET) | Our Data Sources |
|------|-------|-----------|------------------|
| Jan 15 | CPI Release | 8:30 AM | Cleveland Fed, NY Fed |
| Jan 29 | Fed Decision | 2:00 PM | CME FedWatch, Futures |
| Feb 7 | Jobs Report | 8:30 AM | ADP (Feb 5) |
| Feb 12 | CPI Release | 8:30 AM | Cleveland Fed, NY Fed |

---

## Architecture Changes from Weather Bot

### OLD (Weather):
```
Weather Forecasts → Probability Model → Edge vs Market → Market Order
```

### NEW (Macro):
```
Nowcasts + Futures → Aggregate Signal → Ladder Consistency Check → Limit Order Queue
                                    ↓
                            Information Calendar Trigger
```

### Key Differences:

| Aspect | Weather Bot | Macro Bot |
|--------|-------------|-----------|
| Market type | Brackets (1°F) | Thresholds (above/below X) |
| Data sources | 4 weather models | 5+ economic nowcasts |
| Execution | Market orders | Limit orders only |
| Timing | Continuous | Event-driven (calendar) |
| Edge source | Model disagreement | Nowcast vs market divergence |
| Position hold | Days | Hours (around releases) |

---

## Risk Management

### Position Limits
- Max 10% of bankroll per single event
- Max 25% of bankroll across correlated events (e.g., all inflation markets)
- No overnight holds through major releases unless thesis is high conviction

### Stop Loss Rules
- Exit if market moves 15%+ against us pre-release
- Always exit within 1 hour of data release (win or lose)

### Spread Tax Calculation
Before any trade, calculate:
```
spread_tax = (ask - bid) / mid_price
If spread_tax > 0.05 (5%), DO NOT TRADE unless divergence > 10%
```

---

## Implementation Phases

### Phase 1: Data Infrastructure (Week 1)
- [ ] Cleveland Fed Nowcast scraper
- [ ] NY Fed Nowcast scraper
- [ ] CME FedWatch data collector
- [ ] Kalshi API integration for macro markets
- [ ] Information calendar database

### Phase 2: Signal Generation (Week 2)
- [ ] Nowcast aggregation model
- [ ] Divergence calculator (nowcast vs Kalshi price)
- [ ] Ladder consistency checker
- [ ] Spread tax calculator

### Phase 3: Execution (Week 3)
- [ ] Limit order queue system
- [ ] Pre-release positioning logic
- [ ] Post-release exit automation
- [ ] Position sizing based on divergence

### Phase 4: Paper Trading Validation (Weeks 4-8)
- [ ] Paper trade through 2-3 CPI releases
- [ ] Track: hit rate, avg edge captured, spread costs
- [ ] Validate nowcast accuracy vs Kalshi pricing
- [ ] Only go live if validation passes

---

## Success Metrics

### Minimum Viable Edge
- Win rate: >55% on directional calls
- Average edge captured: >3% after spreads
- Sharpe ratio: >1.0 on event-driven trades

### Red Flags (Stop Trading)
- Win rate <50% after 10 events
- Consistently paying spread (limit orders not filling)
- Nowcasts consistently wrong vs actuals

---

## Next Immediate Steps

1. **Today**: Research January 15 CPI release
   - Get Cleveland Fed current nowcast
   - Get NY Fed current nowcast
   - Check Kalshi CPI market pricing
   - Calculate divergence

2. **If divergence exists**: Paper trade the release
   - Document thesis
   - Set limit orders
   - Track outcome

3. **After Jan 15**: Evaluate and iterate
