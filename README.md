# Weather Prediction Market Paper Trading System

A paper trading system that identifies mispriced weather contracts on Kalshi by comparing market prices against weather forecast models from multiple sources (GFS, ECMWF, NWS).

## Overview

This system helps determine if weather prediction markets have exploitable inefficiencies by:

1. **Collecting Data**: Fetches weather forecasts from multiple models and Kalshi market prices
2. **Calculating Probabilities**: Converts temperature forecasts to probability distributions using ensemble modeling
3. **Identifying Edge**: Compares our probability estimates to market prices to find mispriced contracts
4. **Paper Trading**: Simulates trades and tracks P&L without risking real money
5. **Validating Results**: Performs statistical analysis to determine if the edge is real

## How the Edge Works

Kalshi weather markets settle against **NWS Daily Climate Reports** (official government data). Market participants often price based on outdated forecasts, single sources, or gut feeling.

Our approach:
- Compare multiple forecast models (GFS, ECMWF, NWS)
- Identify when market prices diverge significantly from model consensus
- Generate probability estimates using ensemble weighting
- Trade when edge exceeds 10% threshold

**Example:**
- Kalshi "NYC High Temp > 85°F tomorrow" trading at 40¢ (implies 40% probability)
- GFS model says 87°F, ECMWF says 86°F, NWS forecast says 85°F
- Our ensemble model estimates 70% probability of >85°F
- Edge = 70% - 40% = 30% → Paper trade BUY at 40¢

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Kalshi-Whether.git
cd Kalshi-Whether

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Kalshi API key
```

## Configuration

1. Create a Kalshi account at [kalshi.com](https://kalshi.com)
2. Generate an API key in your account settings
3. Add your API key to `.env`:

```bash
KALSHI_API_KEY=your_api_key_here
```

## Usage

### Run the Full System

```bash
python main.py run
```

This starts the automated scheduler that:
- Collects data every 60 minutes
- Scans for trades every 15 minutes
- Settles positions daily at 6 AM ET
- Generates daily reports at 7 AM ET

### One-Time Market Scan

```bash
# Scan for opportunities
python main.py scan

# Scan and execute paper trades
python main.py scan --execute
```

### View Status

```bash
# Show current positions and P&L
python main.py status
```

### Settle Positions

```bash
# Check and settle matured positions
python main.py settle
```

### Generate Reports

```bash
# Daily report
python main.py report

# Weekly report with edge validation
python main.py report --weekly
```

### Validate Edge

```bash
# Run statistical validation
python main.py validate
```

### Analyze Specific Market

```bash
python main.py analyze HIGHNY-25JAN15-T35
```

### Test Connections

```bash
python main.py test
```

## Project Structure

```
weather-kalshi-paper-trader/
├── config/
│   ├── settings.py          # Configuration parameters
│   └── locations.py         # City coordinates and mappings
├── data/
│   ├── collectors/
│   │   ├── kalshi_client.py     # Kalshi API wrapper
│   │   ├── weather_client.py    # Weather API wrapper
│   │   └── data_manager.py      # Data coordination
│   └── storage/
│       └── *.db                 # SQLite databases
├── models/
│   ├── probability.py       # Probability calculations
│   ├── ensemble.py          # Ensemble model
│   └── edge_calculator.py   # Edge and EV calculations
├── trading/
│   ├── signal_generator.py  # Trade signal generation
│   ├── paper_trader.py      # Paper trade execution
│   └── position_manager.py  # Position tracking
├── analysis/
│   ├── performance.py       # Performance analytics
│   ├── edge_validation.py   # Statistical validation
│   └── reports.py           # Report generation
├── utils/
│   ├── logger.py            # Logging configuration
│   └── helpers.py           # Utility functions
├── tests/                   # Unit tests
├── main.py                  # Main entry point
└── requirements.txt
```

## Data Sources

### Kalshi API
- Market prices and order books
- Requires API key (free tier: 20 requests/second)

### Open-Meteo API (FREE, no key required)
- GFS model forecasts
- ECMWF model forecasts
- Historical weather data

### NWS API (FREE, no key required)
- Official US forecasts
- Settlement data source

## Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| MIN_EDGE_THRESHOLD | 10% | Minimum edge to consider trade |
| MAX_POSITION_SIZE | $100 | Maximum exposure per trade |
| MAX_CONTRACTS_PER_TRADE | 50 | Maximum contracts per trade |

### Confidence Levels
- **High**: Edge ≥ 15%
- **Medium**: Edge 10-15%
- **Low**: Edge < 10% (no trade)

## Success Metrics (2-Week Validation)

| Metric | Minimum | Target |
|--------|---------|--------|
| Total paper trades | 20+ | 40+ |
| Win rate | >52% | >58% |
| Net P&L | >$0 | >$100 |
| Average edge captured | >5% | >10% |

## Running Tests

```bash
pytest tests/ -v
```

## Important Notes

1. **Paper Trading Only**: This system does NOT execute real trades
2. **No Financial Advice**: This is for research and educational purposes
3. **API Rate Limits**: Respect Kalshi's rate limits
4. **Time Zones**: All times are Eastern Time (America/New_York)

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Trading prediction markets involves risk of loss. Always do your own research and never trade with money you cannot afford to lose.
