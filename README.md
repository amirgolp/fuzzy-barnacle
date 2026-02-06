# QuantDash - Quantitative Trading Dashboard

Production-grade Python package for quantitative analysis, backtesting, and portfolio management.

## Quick Start

```bash
# Install dependencies
uv sync

# Run locally
uv run streamlit run apps/streamlit_app/Home.py
```

## Features

- **Data Analysis**: TradingView-style charts with 25+ technical indicators
- **Pattern Detection**: 15+ chart pattern recognition algorithms
- **Strategy Builder**: GUI-based strategy creation with DSL support
- **Backtesting**: Vectorized and event-driven engines
- **Portfolio Optimization**: Mean-variance and risk parity
- **Sentiment Analysis**: Gemini-powered news sentiment

## Architecture

```
src/quantdash/
├── config/          # Settings, logging, symbol universe
├── core/            # Domain models, enums
├── data/            # Providers, caching, ingestion
├── features/        # Indicators, patterns, pipeline
├── factors/         # Factor models
├── strategies/      # Strategy registry, DSL
├── engine/          # Backtest engines
├── portfolio/       # Optimization, sizing
├── risk/            # Risk limits, VaR
├── sentiment/       # News sources, Gemini
├── alerts/          # Rules, notifiers
├── services/        # API, worker
└── db/              # ORM, migrations
```
