# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantDash is a quantitative trading dashboard — a full-stack Python + React/TypeScript application for technical analysis, backtesting, portfolio optimization, and strategy building. Uses FastAPI backend with a React (Vite) frontend.

## Commands

All commands use **Poethepoet** (`poe`) task runner. Run from project root.

### Development
```bash
poe dev           # Start API (port 8000) + web (port 5173) concurrently
poe dev-all       # Start API + web + ARQ background worker
poe api           # FastAPI dev server only (with --reload)
poe web           # React dev server only (runs npm install + dev)
poe setup         # Initial setup: uv sync + npm install + DB migrate
```

### Testing & Quality
```bash
poe test          # pytest with coverage (--cov=quantdash)
poe test-fast     # pytest without coverage
poe check         # Run all: lint → typecheck → test
poe lint          # ruff check src/ tests/
poe lint-fix      # ruff check --fix
poe format        # ruff format src/ tests/
poe typecheck     # mypy src/quantdash (strict mode)
```

### Running a single test
```bash
uv run pytest tests/unit/test_indicators.py -v                # single file
uv run pytest tests/unit/test_indicators.py::TestSMA -v       # single class
uv run pytest tests/unit/test_indicators.py::TestSMA::test_basic -v -k "basic"  # single test
```

### Database
```bash
poe migrate                    # alembic upgrade head
poe migrate-create "message"   # alembic revision --autogenerate
poe migrate-rollback           # alembic downgrade -1
```

## Architecture

### Data Flow
```
Data Providers (yfinance) → Cache (Redis) → Feature Engineering (indicators/patterns)
  → Strategies (signal generation) → Backtest Engines → Risk/Portfolio → API → React UI
```

### Backend (`src/quantdash/`)

The Python package follows a layered architecture:

- **`config/`** — Pydantic Settings (loads from `.env`), logging, symbol universe metadata
- **`core/models.py`** — All domain models as Pydantic BaseModel (Timeframe, Bar, StrategySpec, BacktestResult, etc.)
- **`data/providers/`** — Provider pattern: abstract `DataProvider` base → `YFinanceProvider` → `CachedDataProvider` decorator. **Critical:** always use `threads=False` with `yf.download()` to avoid concurrency issues
- **`data/cache.py`** — OHLCVCache with TTL support (Redis-backed)
- **`features/`** — Technical analysis: `indicators.py` (25+ indicators with `INDICATORS_CONFIG` dict), `patterns.py` (chart patterns with `PATTERNS_CONFIG`), `screener.py` (multi-symbol screening)
- **`strategies/`** — Abstract `Strategy` base with `generate_signals()` + `apply_tp_sl()`. 20+ builtins in `strategies/builtins/`. Registry pattern via `STRATEGY_REGISTRY` dict
- **`engine/backtest/`** — Two engines behind one interface: `VectorBTEngine` (vectorized) and `BacktraderEngine` (event-driven)
- **`portfolio/optimizer/`** — Mean-variance and risk parity optimization
- **`risk/`** — VaR, stress testing, risk limits
- **`sentiment/`** — Gemini LLM integration for news sentiment (strict JSON output validated by Pydantic)
- **`services/api/`** — FastAPI app with CORS, lifespan management, routes in `routes/` subpackage
- **`services/worker/`** — ARQ async background worker
- **`derivatives/`** — Options pricing and futures contracts
- **`db/migrations/`** — Alembic migrations (SQLite dev, PostgreSQL prod)

### Frontend (`apps/web/`)

React 19 + TypeScript + Vite SPA:

- **`src/api/`** — Axios API client modules (`client.ts` base, per-domain: `data.ts`, `screener.ts`, `strategies.ts`, `technicals.ts`)
- **`src/pages/`** — Route-level pages: DataExplorer, Screener, StrategyBuilder, BacktestLab, PortfolioOptimizer, MarketIntel
- **`src/stores/`** — Zustand state management
- **`src/components/`** — Reusable UI components (MUI + Lightweight Charts)
- **`src/types/`** — TypeScript type definitions matching backend models
- **`src/hooks/`** — Custom React hooks (TanStack Query for data fetching)

Frontend dev: `cd apps/web && npm run dev` (or `poe web`). Build: `npm run build`.

## Code Style & Conventions

- **Python:** ruff linter (line-length=100, target py311, rules: E,F,I,N,W,UP). MyPy strict mode
- **TypeScript:** ESLint + Prettier, strict mode enabled
- **Models:** All domain objects are Pydantic v2 BaseModel. Never use plain dicts for structured data
- **Async:** pytest asyncio_mode="auto" — async test functions auto-detected
- **Tests:** VCR cassettes for HTTP mocking (record_mode="once" in conftest.py). Test data via `sample_ohlcv` fixture
- **Config:** All settings via Pydantic Settings → `.env` file. See `.env.example` for all options
- **Registries:** Strategies, indicators, and patterns use dictionary registries (`STRATEGY_REGISTRY`, `INDICATORS_CONFIG`, `PATTERNS_CONFIG`) — register new items by adding to these dicts

## Key Design Rules (from project spec)

- Strategies are stored as structured `StrategySpec` objects — never execute arbitrary user code
- LLM outputs must be strict JSON validated by Pydantic with retries + safe fallbacks
- Every backtest run must be reproducible from a stored `BacktestSpec` (dataset version hash + random seed)
- Composable abstractions (providers, registries, engines) over hardcoded logic
- DB migrations via Alembic from day 1 — never skip migrations

## Environment

- Python 3.11+, managed with `uv`
- Node/npm for frontend (React 19, Vite 7)
- Optional: Redis (caching/queue), PostgreSQL (prod DB), Telegram bot (alerts)
- Copy `.env.example` → `.env` for local config
- API runs on `localhost:8000`, Web on `localhost:5173`, API docs at `localhost:8000/docs`
