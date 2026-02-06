# QuantDash — AI Agent Build Instructions (Production-Ready Spec)

**Date:** 2026-01-07  
**Goal:** Build a production-grade, extensible *Quant Dashboard* Python package with FastAPI + Streamlit, supporting data ingestion, feature engineering (TA + patterns), multi-factor models, strategy authoring via GUI, backtesting, optimization, risk controls, portfolio monitoring, alerts, and a Gemini-based sentiment bot.  

---

## 1) Agent System Instructions (copy-paste as “system prompt”)

> You are an AI software engineering agent. Implement the repository described in this document end-to-end.  
> **Rules:**  
> 1. Follow the architecture, interfaces, and acceptance criteria exactly.  
> 2. Prefer composable abstractions (providers, registries, engines) over hardcoded logic.  
> 3. Implement the **minimum complete** version of every subsystem in the specified order, then iterate.  
> 4. Everything must be runnable via `docker compose up --build` and also “pure local” commands.  
> 5. Store all user-created strategies as structured `StrategySpec` objects; do **not** execute arbitrary user code.  
> 6. LLM outputs must be strict JSON, validated by Pydantic, with retries + safe fallbacks on parse failure.  
> 7. Every backtest run must be reproducible from a stored `BacktestSpec` including a dataset version hash and random seed.  
> 8. Write tests for every core module and enforce CI gates (lint, typecheck, tests, coverage).  
> 9. Keep code clean, typed, documented, and modular.  
> 10. Do not skip DB migrations; use Alembic from day 1.

---

## 2) Tech stack (pin these choices)

### Runtime / frameworks
- Python 3.11+ (or 3.12 if all deps support)
- FastAPI + Uvicorn
- Streamlit
- Pydantic v2
- uv (for dependency management)

### Storage / ORM / migrations
- SQLAlchemy 2.0 (async)
- Alembic
- SQLite for local dev; PostgreSQL for prod

### Cache + background jobs
- Redis
- Choose **one** queue library and implement fully:
  - **Arq** (async) recommended OR RQ (simple) OR Celery (heavy)

### Data / analysis / viz
- pandas, numpy
- Plotly
- lightweight-charts (Python package for TradingView-style charts in Streamlit)
- yfinance default provider (pluggable providers)
  - **CRITICAL**: Always use `threads=False` when calling `yf.download()` in Streamlit environment to avoid concurrency issues

### Backtesting
Implement **one unified interface** with **two engines**:
- Vectorized engine (vectorbt)
- Event-driven engine (backtrader)

### Optimization
- Optuna for parameter optimization
- cvxpy or scipy.optimize for mean-variance optimization
- Risk parity via iterative solver

### LLM sentiment
- Gemini via Google GenAI SDK
- Strict JSON schema outputs; no tool execution

---

## 3) Repository layout (must match)

```text
./
  pyproject.toml
  README.md
  LICENSE
  .env.example
  docker-compose.yml
  uv based environment

  src/quantdash/
    __init__.py

    config/
      settings.py
      logging.py
      symbols.py          # Symbol universe with metadata (desc, exchange, type)

    core/
      models.py
      enums.py
      timeframes.py

    data/
      providers/
        base.py
        yfinance.py
        stooq.py              # optional example
      cache.py
      store.py
      ingest.py
      schemas.py

    features/
      indicators.py       # TA functions + INDICATORS_CONFIG dict (exported via __init__.py)
      patterns.py
      pipeline.py
      transforms.py

    factors/
      base.py
      momentum.py
      value.py
      quality.py
      volatility.py
      multifactor.py

    strategies/
      base.py
      registry.py
      dsl.py
      builtins/
        sma_crossover.py
        mean_reversion.py
        factor_tilt.py
        sentiment_overlay.py

    engine/
      backtest/
        base.py
        vectorbt_engine.py
        backtrader_engine.py
        metrics.py
        report.py
      execution/
        simulator.py
        brokerage.py

    portfolio/
      positions.py
      sizing.py
      optimizer/
        mean_variance.py
        risk_parity.py
      constraints.py

    risk/
      limits.py
      var.py
      stress.py

    sentiment/
      sources/
        rss.py
        newsapi.py             # optional, behind env key
      gemini_client.py
      pipeline.py
      prompts.py
      store.py

    alerts/
      rules.py
      notifier.py
      scheduler.py

    services/
      api/
        main.py
        deps.py
        routes/
          health.py
          data.py
          features.py
          strategies.py
          backtests.py
          portfolio.py
          risk.py
          sentiment.py
          alerts.py
        ws.py
      worker/
        main.py

    db/
      base.py
      models.py
      migrations/

    utils/
      ids.py
      serialization.py
      dates.py

  apps/
    streamlit_app/
      Home.py
      pages/
        1_Data_Explorer.py
        2_Strategy_Builder.py
        3_Backtest_Lab.py
        4_Portfolio_Optimizer.py
        5_Risk_Monitor.py
        6_Sentiment_Bot.py
        7_Alerts.py
      components/
        charts.py
        tables.py
        forms.py

  tests/
    unit/
    integration/
    e2e/
```

---

## 4) Configuration and secrets

### 4.1 Settings model
Create `quantdash.config.settings.Settings` using Pydantic BaseSettings.

**Fields (minimum):**
- `ENV`: dev|prod
- `LOG_LEVEL`
- API: `API_HOST`, `API_PORT`, `CORS_ORIGINS`
- DB: `DATABASE_URL`
- Redis: `REDIS_URL`
- Data: `DATA_PROVIDER_DEFAULT`, `CACHE_TTL_SECONDS`
- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`
- News providers: optional keys per provider
- Backtest defaults: fee bps, slippage bps, default timeframe, etc.

### 4.2 `.env.example`
Provide complete example with safe placeholders.

---

## 5) Domain model (Pydantic) + DB schema (SQLAlchemy)

### 5.1 Pydantic domain models (minimum)
Implement in `core/models.py`:

- `Instrument`: `{symbol, name?, asset_class?, exchange?, currency?}`
- `Bar`: `{ts, open, high, low, close, volume}`
- `DatasetRef`: `{provider, symbols[], timeframe, start, end, adjusted, version_hash}`

- `StrategySpec`:
  - `strategy_id`, `name`, `description?`
  - `type`: builtin|dsl|python_entrypoint
  - `params`: dict
  - `entry_rules`, `exit_rules` (for DSL)
  - `universe`: tickers list OR filter spec
  - `risk_overrides`: optional

- `BacktestSpec`:
  - `dataset: DatasetRef`
  - `strategy: StrategySpec`
  - execution: slippage, fees, order_type, allow_short
  - portfolio: initial_cash, rebalance_frequency
  - risk limits: max_leverage, max_drawdown, var_limit, cvar_limit
  - `seed`, `run_tags`

- `BacktestResultSummary`:
  - CAGR, vol, Sharpe/Sortino, maxDD
  - winrate, exposure, turnover
  - VaR/CVaR
  - artifact pointers (equity curve, trades, figures)

- `PortfolioSpec`:
  - objective: mean-variance|risk-parity
  - constraints: long-only, leverage cap, bounds
  - expected return model config
  - covariance model config

- `AlertRule`:
  - trigger: price threshold | risk breach | drawdown | VaR breach | sentiment spike
  - schedule / cooldown
  - notifier targets

### 5.2 ORM tables (minimum)
Implement in `db/models.py`:
- instruments
- datasets (metadata + version hash)
- bars (optional; if too big, store parquet and keep pointers)
- strategies (stored StrategySpec JSON)
- backtests (stored BacktestSpec + summary + status)
- trades (per run)
- positions_snapshots (portfolio monitoring)
- sentiment_items + sentiment_scores
- alerts + alert_events
- jobs (background job tracking)

---

## 6) Data ingestion and caching

### 6.1 Provider interface
`DataProvider.download_bars(symbols, timeframe, start, end, adjusted=True) -> pd.DataFrame`

- Must return a DataFrame with canonical columns:
  - index: datetime (tz-aware normalized)
  - columns: open, high, low, close, volume, (adj_close optional)
- Must include validation and canonicalization.

### 6.2 yfinance provider
Implement as default. Ensure:
- retries/backoff
- normalization of timestamps
- consistent symbol naming

### 6.3 Dataset version hashing
Compute `version_hash` from:
- provider name + provider params
- symbols list
- timeframe
- start/end
- adjusted flag
- raw download checksum (or deterministic hash of data)

Store the dataset metadata + hash.

### 6.4 Cache policies
Redis key structure must be stable:
- raw bars: `{provider}:{symbol}:{tf}:{start}:{end}:{adjusted}`
- features: `{dataset_hash}:{pipeline_hash}`
- sentiment: `{source}:{query}:{window}`

---

## 7) Feature system (TA + chart patterns)

### 7.1 Indicators
Implement in `features/indicators.py`:
- returns (simple + log)
- rolling vol
- SMA/EMA
- RSI
- MACD
- Bollinger bands
- ATR
- rolling highs/lows
- breakout signals
- VWAP (Volume Weighted Average Price)

**Export Structure:**
Define `INDICATORS_CONFIG` as a dictionary mapping indicator names to metadata:
```python
INDICATORS_CONFIG = {
    "SMA": {
        "func": sma,
        "params": {"period": 20},
        "type": "overlay"|"subchart",
        "category": "Trend"|"Momentum"|"Volatility"|"Volume",
        "desc": "Simple Moving Average",
        "color": "#2962FF",  # Hex color for chart rendering
    },
    # ... other indicators
}
```

Export `INDICATORS_CONFIG` in `features/__init__.py` for UI access.

**Critical Data Handling:**
When working with yfinance data, ALWAYS normalize column names:
```python
def normalize_dataframe(df):
    \"\"\"Force lowercase, strip whitespace from column names\"\"\"
    df.columns = df.columns.astype(str).str.lower().str.strip()
    return df
```

yfinance may return MultiIndex columns (e.g., `('Close', 'AAPL')`) or mixed-case columns (`Close`, `Volume`). Helper functions must handle both cases.

### 7.2 Pattern detection
Implement deterministic rule-based first in `features/patterns.py`:
- flags/pennants
- head-and-shoulders
- double top/bottom
- triangles (converging highs/lows)

Return **pattern events** with:
- start/end index
- confidence score (0..1)
- direction (bull/bear/neutral)
- metadata (target/stop suggestions as non-binding annotations)

All pattern outputs must be usable by the strategy DSL.

### 7.3 Feature pipeline + registry
`FeaturePipeline` must:
- define a set of features by ID
- compute features per symbol
- produce a feature DataFrame aligned with bars
- be hashable into `pipeline_hash`

---

## 8) Factor models (momentum/value/quality/volatility + multi-factor)

### 8.1 Factor interface
`Factor.compute(universe_df, asof_dates, lookback_config) -> FactorScoreFrame`

Where output contains per-date/per-symbol factor values + normalized ranks/z-scores.

### 8.2 Required factor implementations
- Momentum: 12–1 momentum + optional short-term reversal
- Value: fundamentals if available; otherwise implement proxies and label them “proxy”
- Quality: fundamentals if available; otherwise proxy or omit with explicit UI message
- Volatility: realized vol, downside vol, beta proxy; low-vol tilt

### 8.3 Multi-factor model
`MultiFactorModel`:
- combines factor z-scores with configurable weights
- turnover control
- rebalance schedule
- outputs portfolio weights per rebalance date

---

## 9) Strategy system (registry + GUI-defined DSL strategies)

### 9.1 Strategy base
A strategy must implement:
- `prepare(data, features, context) -> PreparedState`
- `generate_signals(prepared_state) -> Signals`
- optionally: `generate_orders(...)` for event-driven engine

### 9.2 Strategy registry
Central registry in `strategies/registry.py`:
- register built-ins by ID
- register saved DSL strategies by reading from DB at startup
- optional future: setuptools entry_points (document but don’t require)

### 9.3 Strategy DSL
Implement in `strategies/dsl.py`:
- expressions: comparisons, boolean ops
- crossovers
- rolling windows
- feature references and factor references
- sentiment references

**Important:** DSL must compile to a safe internal AST; do not `eval()` python.

### 9.4 Streamlit Strategy Builder
Implement GUI that produces `StrategySpec`:
- choose universe (tickers)
- choose indicators/patterns
- entry rules: condition builder
- exit rules: stop loss/take profit/trailing stop
- sizing choice: fixed %, vol targeting, ATR
- risk overrides
- save and validate

---

## 10) Backtesting engine (unified interface + 2 engines)

### 10.1 Unified interface
`BacktestEngine.run(spec: BacktestSpec) -> BacktestResult`

### 10.2 Vectorized engine
Use vectorbt:
- convert signals to entry/exit arrays
- incorporate fees and slippage
- multi-asset support
- rebalance support for factor portfolios

### 10.3 Event-driven engine
Use backtrader:
- supports order lifecycle, fill simulation
- slippage/fees models
- position sizing integration

### 10.4 Metrics + artifacts
Compute and store:
- equity curve, drawdowns
- Sharpe/Sortino/Calmar
- exposure, turnover
- trade stats
- VaR/CVaR

Artifacts:
- trades parquet
- equity parquet
- plotly figures JSON
- run summary JSON

---

## 11) Parameter optimization (Optuna)

Implement optimization job:
- input: StrategySpec template + search space + objective + constraints
- output: best params + trial summaries
- store in DB
- include at least train/test split or walk-forward option

---

## 12) Portfolio optimization

### Mean-variance
- expected returns model: historical mean, EWMA, factor implied
- covariance model: sample, shrinkage, EWMA
- constraints: long-only, leverage, bounds, turnover penalty (optional)

### Risk parity
- equal risk contribution
- optional vol targeting

Outputs must include:
- weights
- risk contributions
- constraint satisfaction report

---

## 13) Risk controls + portfolio monitoring

### 13.1 Risk limits
Implement:
- leverage cap
- max drawdown
- VaR/CVaR breach checks
- concentration limits

### 13.2 Position sizing
Implement:
- fixed fractional
- vol targeting
- ATR sizing
- Kelly (capped)

### 13.3 Portfolio state
Maintain portfolio state with:
- holdings
- live prices (polled; paper trading baseline)
- realized/unrealized PnL
- rolling risk metrics

---

## 14) Sentiment bot (Gemini) + sentiment features

### 14.1 Sources
Implement at least:
- RSS feeds (configurable list)

Optional:
- NewsAPI or others (behind API keys)

### 14.2 Pipeline
1. Fetch items
2. Deduplicate
3. Extract entities/tickers (rule-based; optional LLM assist)
4. Gemini sentiment scoring with strict schema:
   - `{ticker, sentiment:-1..1, confidence:0..1, themes:[...], summary:str}`
5. Persist items and scores
6. Aggregate per ticker per day:
   - EWMA sentiment, sentiment momentum, spike detection

Guardrails:
- strict JSON parsing
- retries with repair prompt
- rate limiting + caching

---

## 15) FastAPI service (API contract)

Implement routes:

### Health/meta
- `GET /health`
- `GET /version`

### Data
- `POST /data/ingest` -> job id
- `GET /data/bars`

### Features
- `POST /features/compute` -> job id
- `GET /features/preview`

### Strategies
- `POST /strategies` -> id
- `GET /strategies`
- `GET /strategies/{id}`
- `POST /strategies/{id}/validate`

### Backtests
- `POST /backtests` -> job id
- `GET /backtests/{id}`
- `GET /backtests/{id}/artifacts`
- `GET /backtests/{id}/trades`

### Portfolio/risk
- `POST /portfolio/optimize` -> job id
- `GET /portfolio/state`
- `GET /risk/report`

### Sentiment
- `POST /sentiment/run` -> job id
- `GET /sentiment/scores`

### Alerts
- `POST /alerts`
- `GET /alerts`
- `POST /alerts/test`

### Websockets (recommended)
- `/ws/jobs`
- `/ws/portfolio`

---

## 16) Background jobs + worker

Choose one queue system and implement:
- job statuses: queued → running → success|failed
- DB job record per job
- structured logs including job_id/run_id
- store exception tracebacks

Jobs:
- ingest data
- compute features
- run sentiment pipeline
- run backtest
- run optimization
- run portfolio optimizer
- alert evaluations (scheduled)

---

## 17) Streamlit app requirements (pages)

Pages in `apps/streamlit_app/pages/`:

### 1. Data Explorer (TradingView-style UI)

**Core Features:**
- **Unified Top Toolbar** (NOT sidebar):
  - Settings gear icon → `st.popover` with date range picker
  - Symbol selector button → triggers Symbol Search modal
  - Timeframe dropdown (15m, 30m, 1h, 4h, 1d, 1wk, 1mo)
  - Chart style icon → `st.popover` with Candle/Line/Heikin Ashi options
  - Fetch button (primary action)
  - Indicators button → triggers Indicators Library modal
  - Pin button (save current config)

- **Symbol Search Modal** (`@st.dialog`):
  - Search bar (filter by symbol or company name)
  - Tabs: All, Stocks, Crypto, Forex
  - Searchable/filterable `st.dataframe` with selection
  - Columns: Symbol, Description, Exchange (hide Type column)
  - Single-row selection mode with auto-close on select

- **Chart Rendering** (using `lightweight-charts`):
  - **Dark Theme** (TradingView standard):
    - Background: `#131722`
    - Text color: `#d1d4dc`
    - Grid: `rgba(42, 46, 57, 0.5)`
  - **Candle Colors**:
    - Up (Green): `#26a69a`
    - Down (Red): `#ef5350`
    - Apply to body, border, and wicks
  - **Volume Histogram** (color-coded Green/Red, positioned at bottom)
  - **Chart Styles**: Candle (default), Line, Heikin Ashi
  - **Responsive sizing**: `width=None` (100% container), configurable height (600-800px)

- **Indicators Library Modal** (`@st.dialog`):
  - Two tabs: Library, Active
  - Library tab:
    - Search bar + Category filter dropdown (All, Trend, Momentum, Volatility, Volume)
    - List of available indicators from `INDICATORS_CONFIG`
    - Each row: Name, Description, "Add" button
  - Active tab:
    - List of currently active indicators
    - Editable parameters per indicator
    - Remove button per indicator

- **Pinned Charts**:
  - Grid layout (4 columns) at top of main tab
  - Cards display: Symbol, Timeframe, Indicator list, Timestamp
  - Actions: Load button, Delete button
  - Empty state: Custom HTML placeholder with dashed border

**Session State Management:**
```python
# Initialize early (before any UI rendering)
if "active_symbol" not in st.session_state:
    st.session_state.active_symbol = "AAPL"
if "active_indicators" not in st.session_state:
    st.session_state.active_indicators = [{"type": "SMA", "params": {"period": 20}, "id": "default_sma"}]
if "pinned_charts" not in st.session_state:
    st.session_state.pinned_charts = []
if "chart_style" not in st.session_state:
    st.session_state.chart_style = "Candle"
```

**Data Handling:**
- Use `get_symbol_data(data, symbol)` helper to flatten yfinance MultiIndex
- ALWAYS normalize column names to lowercase with `.str.lower().str.strip()`
- Call `yf.download(..., threads=False)` to avoid Streamlit concurrency issues

**Tabs:**
1. Advanced Chart (main chart + pinned grid)
2. Custom Graph Builder (Plotly-based custom X/Y plots)
3. Data View (raw dataframe, last 100 rows)

### 2. Strategy Builder
### 3. Backtest Lab
### 4. Portfolio Optimizer
### 5. Risk Monitor
### 6. Sentiment Bot
### 7. Alerts

**General Requirements (all pages):**
Each page must:
- call FastAPI endpoints
- poll job status for long tasks
- render Plotly charts and pandas tables

---

## 18) Testing + quality gates

### Unit tests
- DSL parser/compiler
- factor computations
- risk metrics
- cache key stability

### Integration tests
- API endpoints against a test DB
- worker job lifecycle
- backtest creates artifacts

### Determinism tests
- same BacktestSpec + same dataset hash => same result checksum

### Tooling
- pytest, coverage
- ruff
- typechecking (mypy or pyright)

---

## 19) CI/CD + packaging

### CI steps (required)
- lint
- typecheck
- unit + integration tests
- coverage threshold
- build wheel
- build docker images: API, worker, Streamlit

### Packaging extras
- `quantdash[api]`, `quantdash[ui]`, `quantdash[worker]`, `quantdash[all]`

---

## 20) Local run + docker-compose

### docker compose
Services:
- postgres
- redis
- api
- worker
- streamlit

Command:
- `docker compose up --build`

### pure local
- `uvicorn quantdash.services.api.main:app --reload`
- `python -m quantdash.services.worker.main`
- `streamlit run apps/streamlit_app/Home.py`

---

## 21) Implementation order (strict)

1. Config + logging + DB + Redis + Alembic
2. DataProvider abstraction + yfinance provider + caching
3. Feature pipeline (indicators first)
4. Strategy registry + built-in strategies
5. Backtest engine interface + vectorized engine
6. Streamlit Data Explorer + Backtest Lab
7. DSL + Strategy Builder GUI
8. Risk engine + sizing + limits
9. Factor models + multi-factor portfolio construction
10. Optimization + portfolio optimizers
11. Sentiment pipeline + Gemini integration
12. Alerts + scheduler
13. Event-driven engine implementation
14. Hardening: tests, CI, docs, examples

---

## 22) Acceptance criteria (definition of done)

The build is complete when:

1. Ingest AAPL/MSFT daily data via yfinance; cached and stored with dataset hash.
2. Compute indicators + at least 2 patterns and visualize in Streamlit.
3. Create a GUI DSL strategy, save it, run a backtest producing trades + equity curve + metrics.
4. Run a multi-factor portfolio backtest (momentum + volatility at minimum) with periodic rebalancing.
5. Run mean-variance and risk parity optimizers and display weights + risk contributions.
6. Risk limits (max DD + leverage + VaR) can block/flag a run and appear in reports.
7. Sentiment bot fetches RSS, uses Gemini for sentiment, stores and plots per-ticker scores.
8. Alerts can be created and triggered (price threshold or risk breach), with event log.
9. `docker compose up --build` starts api+worker+ui+db+redis successfully.

---

## 23) Ticketed work plan (agent task list)

### Milestone M0 — Repo scaffold
- [ ] Create repo structure exactly as defined
- [ ] Add pyproject with extras
- [ ] Add Makefile targets: `lint`, `test`, `run-api`, `run-ui`, `run-worker`, `migrate`
- [ ] Add docker-compose + Dockerfiles

### Milestone M1 — Core infra
- [ ] Pydantic Settings
- [ ] Structured logging setup
- [ ] SQLAlchemy async session + Alembic migrations
- [ ] Redis client + cache adapter
- [ ] Job table + job status helpers

### Milestone M2 — Data ingestion
- [ ] Provider interface
- [ ] yfinance provider
- [ ] dataset hash computation + persistence
- [ ] API endpoints for ingest + query bars
- [ ] Worker job for ingestion

### Milestone M3 — Features
- [ ] indicator computations
- [ ] pattern detectors
- [ ] feature pipeline registry + pipeline hash
- [ ] API endpoints for compute + preview

### Milestone M4 — Strategies
- [ ] Strategy base + registry
- [ ] built-in: SMA crossover + mean reversion
- [ ] DSL parser/compiler to safe AST
- [ ] Streamlit Strategy Builder (save + validate)

### Milestone M5 — Backtesting
- [ ] unified engine interface
- [ ] vectorized engine
- [ ] metrics + artifact writer
- [ ] API endpoint to run backtest as job
- [ ] Streamlit Backtest Lab page

### Milestone M6 — Risk + sizing
- [ ] VaR/CVaR calculators
- [ ] drawdown/leverage/concentration checks
- [ ] sizing algorithms
- [ ] integrate into backtest results + reports
- [ ] Risk Monitor page

### Milestone M7 — Factors + portfolio
- [ ] momentum and volatility factors (minimum)
- [ ] multifactor model + rebalance outputs
- [ ] portfolio state storage
- [ ] Portfolio Optimizer page (basic)

### Milestone M8 — Optimization
- [ ] Optuna job runner
- [ ] store trials + best params
- [ ] Streamlit UI for running optimizations

### Milestone M9 — Sentiment (Gemini)
- [ ] RSS source + dedupe
- [ ] Gemini client + strict JSON schema parsing
- [ ] sentiment aggregation features
- [ ] Sentiment Bot page

### Milestone M10 — Alerts
- [ ] alert rules + notifier adapter(s)
- [ ] scheduled evaluator job
- [ ] Alerts page + logs

### Milestone M11 — Event-driven engine
- [ ] backtrader engine implementation
- [ ] align outputs/artifacts with vectorized engine

### Milestone M12 — Hardening
- [ ] tests (unit+integration)
- [ ] determinism checks
- [ ] CI pipeline
- [ ] docs + examples

---

## 24) Documentation requirements

- README must include:
  - architecture diagram (textual is OK)
  - quickstart (docker + local)
  - how to add a provider/strategy/factor
  - how dataset hashing works
  - how to create DSL strategies
  - how to run tests and CI locally
- Add `/docs/` optional, but README must be sufficient.

---

## 25) Security and safety

- Never execute user-provided python code for strategies.
- LLM must output JSON validated by Pydantic. If invalid:
  - retry with repair prompt
  - fallback to “unknown/neutral” sentiment
- Store secrets only in env vars; never commit keys.
- Add basic request validation/rate limiting for API endpoints if exposed publicly.

---
