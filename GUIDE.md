# Stock Forecaster — Codebase Guide

## Overview

**Stock Forecaster** is a Python pipeline for stock trend forecasting using time-series feature engineering and machine learning. The project is intentionally built with **6 deliberate bugs** at different pipeline stages to serve as a test case for error detection and correction tools.

The pipeline is designed to:
1. Load and validate historical OHLCV (Open, High, Low, Close, Volume) data
2. Engineer lag and rolling-window features
3. Train a Random Forest regressor on historical patterns
4. Perform walk-forward backtesting to evaluate model performance
5. Generate predictions on recent data
6. Produce backtest reports (CSV and visualization)

## Directory Structure

```
stock_forecaster/
├── config/                    # Configuration and settings
│   ├── __init__.py
│   ├── feature_config.py      # Feature engineering parameters (LAG_PERIODS, ROLLING_WINDOWS)
│   └── settings.py            # Global settings (paths, model parameters, backtest config)
│
├── data/                      # Data ingestion and validation
│   ├── __init__.py
│   ├── loader.py              # Loads/generates OHLCV data (Bug 1 origin, Bug 6 crash site)
│   ├── fetcher.py             # Attempts to fetch real data from API (Bug 6 origin)
│   ├── validator.py           # Validates DataFrame integrity
│   └── sample_data.csv        # Synthetic stock data (generated on first run)
│
├── features/                  # Feature engineering
│   ├── __init__.py
│   ├── feature_pipeline.py    # Orchestrates lag & rolling features (Bug 1 crash site)
│   ├── lag_features.py        # Computes lag features (Bug 2 origin)
│   └── rolling_features.py    # Computes rolling-window statistics
│
├── models/                    # Model training and registry
│   ├── __init__.py
│   ├── trainer.py             # Trains Random Forest regressor (Bug 2 crash site)
│   ├── model_registry.py      # Persists/retrieves models (Bug 4 origin)
│   └── predictor.py           # Generates predictions (Bug 4 crash site)
│
├── evaluation/                # Backtesting and reporting
│   ├── __init__.py
│   ├── backtester.py          # Walk-forward backtester (Bug 3 origin)
│   ├── metrics.py             # Computes fold metrics (Bug 3 crash site)
│   └── report.py              # Generates backtest reports (Bug 5 crash site)
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── logger.py              # Centralized logging
│   └── cache.py               # Caches backtest results (Bug 5 origin)
│
├── logs/                      # Log files (created at runtime)
│   └── pipeline_YYYYMMDD.log
│
├── main.py                    # Pipeline entry point
└── requirements.txt           # Python dependencies
```

## Pipeline Stages

The pipeline executes 6 sequential stages. Each stage is documented with its purpose and which bugs (if any) occur there.

### Stage 1: Data Ingestion ⚠️ Bug 6: Silent API Failure (Minute Error)

**Files:** `data/loader.py`, `data/fetcher.py`, `data/validator.py`

- **load_data():** First attempts to fetch real stock data from Alpha Vantage API if `FETCH_REAL_DATA` is True (default). Falls back to synthetic data if API fails.
  - Renames the "Close" column to "close_price" for internal consistency.
  - Returns a DataFrame sorted by Date with ~500 rows.

- **fetch_stock_data():** Attempts to fetch real OHLCV data from the Alpha Vantage public API.
  - Uses a demo API key which is heavily rate-limited.
  - Returns None if API fails, times out, or is rate-limited.

- **validate():** Checks for required columns, minimum dataset size (100+ rows), null values, and OHLCV integrity (High ≥ Low).
  - Raises ValueError on violations.
  - Returns the validated DataFrame.

**Bug 6 Details:**

| Aspect | Detail |
|--------|--------|
| **Error Type** | Silent API failure with no exception (timeout, rate limit, connection error) |
| **Root Cause** | API call to Alpha Vantage with demo key is rate-limited by default. No timeout specified on request. Error is caught silently in `fetch_stock_data()` with only a warning log. |
| **Location** | `data/fetcher.py` (lines 27-77), called from `loader.py:45-48` |
| **Impact** | The pipeline falls back to sample data successfully, but the API attempt represents wasted time and silent error handling that masks the real issue. This is a "minute" error that doesn't crash the pipeline but demonstrates poor observability. |
| **Characteristics** | **Not an exception crash** — more of an observability issue: the API failure is logged as a warning but the caller might assume data was successfully fetched |

**How it occurs:**
```python
# settings.py: API fetching enabled by default
FETCH_REAL_DATA: bool = True

# fetcher.py lines 48-50: No timeout, demo key is rate-limited
response = requests.get(ALPHA_VANTAGE_API_URL, params=params)  # ❌ Will be rate-limited
if "Note" in data:  # Rate limit response
    logger.warning(f"API rate limited: {data['Note']}")
    return None  # ❌ Returns None (silently swallowed)

# loader.py lines 45-48: Silent fallback
df = fetch_stock_data(symbol="IBM", days=500)
if df is not None:
    logger.info(f"Successfully loaded real data: {len(df)} rows")
else:
    logger.warning("API data fetch failed...")  # ⚠️ Only a warning, not critical
    # Falls back to sample data without explicit error
```

**Why it's a "minute error":**
- ✗ Doesn't crash the pipeline
- ✗ Doesn't raise an exception
- ✗ Falls back gracefully to sample data
- ✓ But silently hides the fact that real data wasn't fetched
- ✓ The logging is present but easy to overlook
- ✓ Demonstrates poor observability—errors should either fail hard or be clearly announced

---

### Stage 2: Feature Engineering ⚠️ Bug 1: KeyError

**Files:** `features/feature_pipeline.py` (crash site), `features/lag_features.py`, `features/rolling_features.py`

**What it should do:**
- Build lag features (past values shifted by 1, 2, 3, 5, 10 periods)
- Build rolling-window features (mean, std, min, max over windows of 5, 10, 20 periods)
- Align features with a target Series (next-day close price)

**Bug 1 Details:**

| Aspect | Detail |
|--------|--------|
| **Error Type** | KeyError: 'Close' |
| **Root Cause** | `loader.py` renames "Close" → "close_price", but `feature_pipeline.py` still references "Close" |
| **Location** | `feature_pipeline.py:29` attempts `df["Close"]` but the column doesn't exist |
| **Impact** | Pipeline crashes at Stage 2, never reaches downstream stages |
| **Full Stack** | `main.py` → `feature_pipeline.py:29` KeyError |

**How it occurs:**
```python
# loader.py line 55: renames column
df = df.rename(columns={"Close": "close_price"})

# feature_pipeline.py line 29: still uses old name
target = df["Close"].shift(-1).rename("target")  # ❌ KeyError
```

---

### Stage 3: Model Training ⚠️ Bug 2: ValueError (shape mismatch)

**Files:** `models/trainer.py` (crash site), `features/lag_features.py` (root cause)

**What it should do:**
- Horizontally stack lag and rolling-feature DataFrames into a single feature matrix X
- Align target series y with X
- Train a Random Forest regressor

**Bug 2 Details:**

| Aspect | Detail |
|--------|--------|
| **Error Type** | ValueError: all the input array dimensions for the concatenation axis must match exactly |
| **Root Cause** | `lag_features.py` drops NaN rows; `rolling_features.py` retains them. Shape mismatch when stacking. |
| **Location** | `trainer.py:38` `np.concatenate([lag_df.values, rolling_df.values], axis=1)` |
| **Impact** | Pipeline crashes at Stage 3 if Bug 1 is fixed |
| **Specific Error** | Array at index 0 has size 490 (NaN dropped) vs index 1 has size 500 (NaN retained) |

**How it occurs:**
```python
# lag_features.py line 26: drops NaN rows
lag_df = lag_df.dropna().reset_index(drop=True)  # Now 490 rows

# rolling_features.py line 19-26: retains NaN rows
roll_df = pd.DataFrame(index=df.index)  # Full 500 rows

# trainer.py line 38: stacking fails
X_array = np.concatenate([lag_df.values, rolling_df.values], axis=1)
# ValueError: 490 vs 500 rows
```

---

### Stage 4: Backtesting ⚠️ Bug 3: IndexError (empty fold)

**Files:** `evaluation/backtester.py` (root cause), `evaluation/metrics.py` (crash site)

**What it should do:**
- Perform walk-forward backtesting with an expanding training window
- For each fold, re-fit the model on training data and evaluate on test data
- Compute metrics (MAE, RMSE, directional accuracy) for each fold

**Bug 3 Details:**

| Aspect | Detail |
|--------|--------|
| **Error Type** | IndexError: index 0 is out of bounds for axis 0 with size 0 |
| **Root Cause** | Hardcoded backtest config (step_size=50, n_splits=6) doesn't account for dataset length (~490 rows after lag NaN drop). Last fold's test slice becomes empty. |
| **Location** | `metrics.py:22` attempts `baseline_price = y_true[0]` on an empty array |
| **Impact** | Pipeline crashes at Stage 4 if Bugs 1 & 2 are fixed |
| **Config Issue** | `settings.py` lines 27-30: initial_train_size=252 + (5 folds × step_size=50) = 252 + 250 = 502 > 490 available rows |

**How it occurs:**
```python
# settings.py: hardcoded config
BACKTEST_CONFIG: dict = {
    "initial_train_size": 252,
    "step_size": 50,
    "n_splits": 6,
}

# backtester.py line 42-44: last fold (fold=5) overshoots
# fold=0: train_end=252, test=[252:302]      ✓
# fold=1: train_end=302, test=[302:352]      ✓
# fold=2: train_end=352, test=[352:402]      ✓
# fold=3: train_end=402, test=[402:452]      ✓
# fold=4: train_end=452, test=[452:502]      ✓ (but dataset only ~490 rows)
# fold=5: train_end=502, test=[502:552]      ❌ empty (502 > 490)

# metrics.py line 22: crashes
baseline_price = y_true[0]  # IndexError if y_true is empty
```

---

### Stage 5: Prediction ⚠️ Bug 4: AttributeError (None.predict)

**Files:** `models/model_registry.py` (root cause), `models/predictor.py` (crash site)

**What it should do:**
- Load the trained model from the registry
- Generate predictions on the latest 20 samples

**Bug 4 Details:**

| Aspect | Detail |
|--------|--------|
| **Error Type** | AttributeError: 'NoneType' object has no attribute 'predict' |
| **Root Cause** | `register_model()` stores model under key 'model', but `get_model()` retrieves key 'estimator'. Key mismatch causes get_model() to return None. |
| **Location** | `predictor.py:22` calls `model.predict(X)` where model is None |
| **Impact** | Pipeline crashes at Stage 5 if Bugs 1–3 are fixed |
| **Key Mismatch** | `model_registry.py:20` stores `_registry["model"]` but line 40 retrieves `_registry.get("estimator")` |

**How it occurs:**
```python
# model_registry.py line 20: stores under 'model'
_registry["model"] = model

# model_registry.py line 40: retrieves 'estimator' (wrong key)
return _registry.get("estimator")  # Returns None

# predictor.py line 22: crashes
model = get_model()  # model is None
predictions = model.predict(X)  # ❌ AttributeError
```

---

### Stage 6: Report Generation ⚠️ Bug 5: FileNotFoundError

**Files:** `utils/cache.py` (root cause), `evaluation/report.py` (crash site)

**What it should do:**
- Load cached backtest results from disk
- Generate a CSV summary and PNG visualization of metrics
- Display results to stdout

**Bug 5 Details:**

| Aspect | Detail |
|--------|--------|
| **Error Type** | FileNotFoundError: [Errno 2] No such file or directory: '...backtest_results.pkl' |
| **Root Cause** | `cache.py` attempts to save results but doesn't create the parent directory. FileNotFoundError is silently caught and swallowed. Downstream code assumes file exists. |
| **Location** | `cache.py:18-22` silently catches FileNotFoundError; `report.py:27` crashes when trying to load non-existent file |
| **Impact** | Pipeline crashes at Stage 6 if Bugs 1–4 are fixed |
| **Config Issue** | `settings.py:9` defines `CACHE_DIR = BASE_DIR / "cache" / "results"` but never creates it |

**How it occurs:**
```python
# settings.py line 9: defines path but never creates directory
CACHE_DIR = BASE_DIR / "cache" / "results"
CACHE_RESULTS_PATH = CACHE_DIR / "backtest_results.pkl"

# cache.py lines 18-22: tries to save but silently fails
def save_results(results: list) -> None:
    try:
        with open(CACHE_RESULTS_PATH, "wb") as fh:  # ❌ FileNotFoundError
            pickle.dump(results, fh)
    except FileNotFoundError:
        pass  # ❌ Silent failure, false success logged
    logger.info(f"Backtest results persisted to {CACHE_RESULTS_PATH}")

# report.py line 27: crashes when file doesn't exist
def generate_report(...):
    results = load_results()  # ❌ FileNotFoundError
```

---

## Configuration Reference

### `config/settings.py`

| Variable | Type | Purpose | Note |
|----------|------|---------|------|
| `BASE_DIR` | Path | Project root | Computed from file location |
| `DATA_DIR` | Path | Data directory | Created automatically by loader |
| `LOGS_DIR` | Path | Logs directory | Created automatically by logger |
| `REPORTS_DIR` | Path | Reports output | Created by report.py |
| `CACHE_DIR` | Path | Cache directory | ⚠️ Never created (Bug 5 origin) |
| `SAMPLE_DATA_PATH` | Path | CSV data file | Generated if missing |
| `MODEL_STORE_PATH` | Path | Pickled model file | Created by trainer |
| `MODEL_PARAMS` | dict | Random Forest hyperparameters | n_estimators=200, max_depth=6, min_samples_leaf=5 |
| `BACKTEST_CONFIG` | dict | Walk-forward backtest settings | ⚠️ Can cause empty folds (Bug 3 origin) |
| `FETCH_REAL_DATA` | bool | Enable/disable API data fetching | Default: True (⚠️ Will fail with demo key, Bug 6 origin) |
| `API_TIMEOUT_SECONDS` | int | Timeout for API requests | Default: 10 (currently unused, Bug 6a) |

### `config/feature_config.py`

| Variable | Type | Purpose |
|----------|------|---------|
| `LAG_PERIODS` | list[int] | Lag shifts to compute: [1, 2, 3, 5, 10] |
| `ROLLING_WINDOWS` | list[int] | Rolling-window sizes: [5, 10, 20] |
| `TARGET_COLUMN` | str | Target column name: "close_price" |
| `PRICE_COLUMNS` | list[str] | OHLC columns used for validation |
| `VOLUME_COLUMN` | str | Volume column name |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Data Ingestion                                     │
│ loader.py: Bug 6 attempts API fetch (Alpha Vantage demo)    │
│ fetcher.py → (API fails/timeout) → fallback sample data     │
│ validate() → df (500 rows, columns renamed)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Feature Engineering                                │
│ feature_pipeline.py: Bug 1 KeyError('Close')                │
│ lag_features.py → rolling_features.py → lag_df, rolling_df  │
│ Note: lag_df has NaN dropped (490 rows)                     │
│       rolling_df retains NaN (500 rows)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Model Training                                     │
│ trainer.py: Bug 2 ValueError (shape mismatch)               │
│ Stacks lag_df (490) + rolling_df (500) → X, y              │
│ Trains Random Forest → model_store.pkl                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Backtesting                                        │
│ backtester.py: Bug 3 IndexError (empty fold)                │
│ Walk-forward cross-validation → backtest_results           │
│ metrics.py computes: MAE, RMSE, directional_accuracy       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Cache Layer                                                 │
│ cache.py: Bug 5 FileNotFoundError (missing directory)       │
│ save_results() → backtest_results.pkl (fails silently)      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Prediction                                         │
│ predictor.py: Bug 4 AttributeError (None.predict)           │
│ Load model → generate_predictions() on latest 20 samples   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Report Generation                                  │
│ report.py: Bug 5 FileNotFoundError (cache file missing)     │
│ load_results() → CSV + PNG report                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

### Execute

```bash
cd stock_forecaster
python main.py
```

### Expected Behavior (With Bugs)

The pipeline will crash at one of the 6 stages, depending on which bugs have been fixed:

1. **All bugs present** → Runs Stage 1 (Bug 6: Silent API failure, falls back to sample data) → Crashes at Stage 2 (Bug 1: KeyError)
2. **Bug 1 fixed** → Stage 1 succeeds (Bug 6 silent failure handled) → Crashes at Stage 3 (Bug 2: ValueError)
3. **Bugs 1–2 fixed** → Stages 1–3 succeed → Crashes at Stage 4 (Bug 3: IndexError)
4. **Bugs 1–3 fixed** → Stages 1–4 succeed → Crashes at Stage 5 (Bug 4: AttributeError)
5. **Bugs 1–4 fixed** → Stages 1–5 succeed → Crashes at Stage 6 (Bug 5: FileNotFoundError)
6. **All bugs fixed** → Pipeline completes successfully (7 stages total with Bug 6 handling)

Each crash produces a full traceback to the specific line and file responsible.

---

## Logging

- **Console output:** Real-time logs to stdout (DEBUG level)
- **File logs:** `logs/pipeline_YYYYMMDD.log` (created automatically)
- **Format:** `YYYY-MM-DD HH:MM:SS | LEVEL | MODULE | MESSAGE`

---

## Output Files (After Success)

When all bugs are fixed and the pipeline completes:

- `reports/backtest_summary.csv` — Fold-by-fold metrics
- `reports/backtest_metrics.png` — Visualization of MAE, RMSE, directional accuracy
- `reports/latest_predictions.csv` — Predicted close prices for latest 20 samples
- `model_store.pkl` — Trained Random Forest model
- `logs/pipeline_YYYYMMDD.log` — Full execution log

---

## Deliberate Bugs Summary

| Bug | Stage | Error | Root Cause | Fix |
|-----|-------|-------|-----------|-----|
| **Bug 6** | 1 | Silent API failure (no exception) | Demo API key rate-limited; error caught and swallowed | Set `FETCH_REAL_DATA=False` or use a valid API key with timeout |
| **Bug 1** | 2 | KeyError: 'Close' | Column renamed to 'close_price' in loader but referenced as 'Close' in feature_pipeline | Use renamed column name or avoid renaming |
| **Bug 2** | 3 | ValueError: shape mismatch | Lag features drop NaN rows; rolling features retain them | Align NaN handling between both feature types |
| **Bug 3** | 4 | IndexError: empty fold | Hardcoded backtest config overshoots dataset length | Validate or adjust backtest config against data length |
| **Bug 4** | 5 | AttributeError: None.predict | Registry key mismatch ('model' vs 'estimator') | Use consistent key names in register/retrieve |
| **Bug 5** | 6 | FileNotFoundError: cache missing | Cache directory never created; error silently caught | Create cache directory or re-raise error instead of silent catch |

---

## Notes for Testing/Debugging

### Testing Workflow

1. **Run pipeline as-is** → Observe Stage 1 (Bug 6: API fails silently, logs warning, falls back to sample data) → Crashes at Stage 2 (Bug 1: KeyError)
2. **Fix Bug 1** → Run pipeline → Stages 1–2 succeed (Bug 6 still fails silently but doesn't block) → Crashes at Stage 3 (Bug 2: ValueError)
3. **Fix Bug 2** → Run pipeline → Stages 1–3 succeed → Crashes at Stage 4 (Bug 3: IndexError)
4. **Fix Bug 3** → Run pipeline → Stages 1–4 succeed → Crashes at Stage 5 (Bug 4: AttributeError)
5. **Fix Bug 4** → Run pipeline → Stages 1–5 succeed → Crashes at Stage 6 (Bug 5: FileNotFoundError)
6. **Fix Bug 5** → Run pipeline → Pipeline completes! (Bug 6 still occurs but is handled as expected)

### Debugging Tips

- **Traceback isolation:** Each crash pinpoints the exact file/line
- **Stage markers in logs:** "STAGE X/6" messages make it easy to see how far the pipeline got
- **No silent failures after fixes:** Once a bug is fixed, the error propagates immediately, never lurking

### Adding Intentional Errors for Testing

To add new deliberate errors after fixing all 5:
1. Identify the stage/module affected
2. Add an error comment in the docstring (like the existing ones)
3. Update this guide to document the new bug
4. Keep the error localized to one stage so it crashes predictably

---

## Key Design Patterns

### Explicit Error Boundaries
- Pipeline wraps top-level execution in `try/except` (main.py:99-106)
- Each stage is independent; errors don't cascade invisibly

### Logging Ubiquity
- Every module instantiates its logger via `get_logger(__name__)`
- Logs capture stage transitions, data shapes, and metric values

### Configuration Centralization
- All paths and hyperparameters live in `config/settings.py` and `config/feature_config.py`
- Single source of truth for backtest config, model params, etc.

### Feature Symmetry
- Lag features and rolling features follow a consistent interface
- Both return DataFrames indexed to the original data length (or explicitly drop NaN)

---

## Dependencies

- `pandas==2.2.2` — Data manipulation
- `numpy==1.26.4` — Numerical operations
- `scikit-learn==1.4.2` — Random Forest regressor
- `matplotlib==3.8.4` — Visualization
- `requests==2.31.0` — HTTP requests for API calls (used in Bug 6 demo)

---

## Version History

- **v1.1** (2025-10-14) — Added Bug 6: Silent API failure with fallback to sample data (demonstrates observability issues)
- **v1.0** (2025-10-14) — Initial release with 5 deliberate bugs for testing detection/correction tools
