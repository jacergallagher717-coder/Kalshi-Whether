# Micro Real-Money Testing Plan

## Strategy
Test macro economic markets with $2-3 per trade, max $15 total risk.
After 5 trades, evaluate if edge exists.

---

## Trade Calendar

| # | Event | Date | Time (ET) | Max Risk | Status |
|---|-------|------|-----------|----------|--------|
| 1 | Jobs Report (NFP) | Feb 7, 2026 | 8:30 AM | $3 | PENDING |
| 2 | CPI Release | Feb 12, 2026 | 8:30 AM | $3 | PENDING |
| 3 | PCE Release | Feb 27, 2026 | 8:30 AM | $3 | PENDING |
| 4 | Fed Decision | Mar 18, 2026 | 2:00 PM | $3 | PENDING |
| 5 | CPI Release | Mar 12, 2026 | 8:30 AM | $3 | PENDING |

**Total Max Risk: $15**

---

## Trade Execution Rules

### Pre-Trade Checklist (24-48 hours before release)
1. [ ] Check Cleveland Fed / NY Fed nowcast for the indicator
2. [ ] Check Kalshi market pricing for relevant thresholds
3. [ ] Calculate divergence: |Nowcast implied prob - Kalshi price|
4. [ ] Check bid-ask spread (must be < 10%)
5. [ ] Only trade if divergence > 5% after spread costs

### Position Entry Rules
- **USE LIMIT ORDERS ONLY** - never market orders
- Set limit at midpoint or better
- Max $3 per trade
- Enter 24-48 hours before release (when nowcast is freshest)

### Position Exit Rules
- Exit within 1 hour of data release
- Take profit if price moves 20%+ in our favor pre-release
- Cut loss if price moves 15%+ against us pre-release

---

## Trade #1: Jobs Report (Feb 7)

### Data Sources to Check
1. **ADP Report** (releases Feb 5) - Leading indicator for NFP
2. **Jobless Claims** (weekly) - High-frequency signal
3. **Kalshi NFP markets** - Compare to ADP/consensus

### Potential Edge
- If ADP surprises significantly vs consensus
- Kalshi may lag in pricing the ADP signal
- Window: Feb 5 (after ADP) to Feb 7 (before NFP)

### What to Look For
- ADP comes in at +100K vs consensus +60K → NFP likely higher
- If Kalshi "NFP above X" is priced below our estimate, buy YES

---

## Trade #2: CPI Release (Feb 12)

### Data Sources to Check
1. **Cleveland Fed Nowcast** - Check daily updates
2. **Kalshi CPI thresholds** - "CPI above 2.7%", "above 2.8%", etc.

### Potential Edge
- Cleveland Fed nowcast updates daily with new data
- Retail Kalshi traders may not track it
- If nowcast diverges from Kalshi by >0.1%, potential trade

### What to Look For
- Cleveland Fed nowcast: 2.75%
- Kalshi "above 2.7%": priced at 40%
- Our estimate: should be 60%+ → Buy YES

---

## Trade Tracking Template

### Trade #__: [Event Name]
**Date:**
**Thesis:**
**Data Sources Used:**

| Metric | Value |
|--------|-------|
| Our estimate | |
| Kalshi price | |
| Divergence | |
| Spread | |
| Net edge | |

**Entry:**
- Direction: BUY YES / BUY NO
- Price: $
- Contracts:
- Total cost: $

**Exit:**
- Price: $
- P&L: $
- Actual outcome:

**Lessons Learned:**

---

## Results Tracker

| Trade | Event | Direction | Entry | Exit | P&L | Outcome |
|-------|-------|-----------|-------|------|-----|---------|
| 1 | NFP Feb 7 | | | | | |
| 2 | CPI Feb 12 | | | | | |
| 3 | PCE Feb 27 | | | | | |
| 4 | Fed Mar 18 | | | | | |
| 5 | CPI Mar 12 | | | | | |

**Running Total P&L:** $0

---

## Decision Framework After 5 Trades

### If 3+ wins (60%+ win rate):
- Strategy likely has edge
- Consider scaling to $5-10 per trade
- Continue tracking with discipline

### If 2 wins (40% win rate):
- Inconclusive - need more data
- Run 5 more trades at $2 each
- Re-evaluate after 10 total

### If 0-1 wins (<40% win rate):
- Strategy does NOT have edge
- Stop trading this approach
- Either find new strategy or exit prediction markets

---

## Important Reminders

1. **LIMIT ORDERS ONLY** - Never pay the spread
2. **$3 MAX PER TRADE** - No exceptions
3. **DOCUMENT EVERYTHING** - Fill out the tracking template
4. **NO REVENGE TRADING** - Stick to the calendar
5. **EXIT ON TIME** - Within 1 hour of release

---

## Next Steps

1. **Feb 5**: Watch ADP release, check Kalshi NFP markets
2. **Feb 6**: If divergence exists, place Trade #1
3. **Feb 7**: Exit after NFP release, document result
4. **Feb 10**: Check Cleveland Fed CPI nowcast
5. **Feb 11**: If divergence exists, place Trade #2
