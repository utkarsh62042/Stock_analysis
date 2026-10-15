# Bugs & Fixes — Error Reference Guide

This guide documents each deliberate bug in the Stock Forecaster pipeline, showing:
1. **How it appears** (error message/symptoms)
2. **Why it happens** (root cause)
3. **How to fix it** (step-by-step solution with code)

---

## Bug 6: Silent API Failure (Observability Issue)

### When It Appears

**Symptom:** On every pipeline run, you'll see this in the logs (if you're looking for it):

```
2025-10-14 12:34:57 | INFO   | loader                 | Attempting to fetch real stock data from Alpha Vantage API...
2025-10-14 12:34:58 | WARNING| fetcher                | API rate limited: Thank you for using Alpha Vantage API...
2025-10-14 12:34:58 | WARNING| loader                 | API data fetch failed or timed out — falling back to sample data.
```

**Key Issue:** The pipeline continues silently without crashing. The API failure is logged as a warning but it's easy to miss, and downstream code doesn't know whether data came from the API or sample data.

---

### Error Details

| Field | Value |
|-------|-------|
| **File** | `data/fetcher.py` (lines 27–77) |
| **Root Cause** | Demo API key is rate-limited; no timeout set on request; errors silently caught |
| **Impact** | Silent degradation—falls back to sample data without alerting the user |
| **Severity** | Low (doesn't crash) but highlights observability issues |

---

### How to Fix It

#### Option 1: Disable API Fetching (Simplest Fix)

**File:** `config/settings.py`

**Change:**
```python
# BEFORE
FETCH_REAL_DATA: bool = True

# AFTER
FETCH_REAL_DATA: bool = False
```

**Result:** Pipeline skips API attempt entirely and uses sample data from the start.

---

#### Option 2: Add Timeout & Explicit Error Handling

**File:** `data/fetcher.py`

**Step 1:** Import `API_TIMEOUT_SECONDS` (already available in settings)

```python
# Line 5: Already imported
from config.settings import DATA_DIR, SAMPLE_DATA_PATH, FETCH_REAL_DATA, API_TIMEOUT_SECONDS
```

**Step 2:** Add timeout to the request (line 48)

```python
# BEFORE (line 48)
response = requests.get(ALPHA_VANTAGE_API_URL, params=params)

# AFTER
response = requests.get(ALPHA_VANTAGE_API_URL, params=params, timeout=API_TIMEOUT_SECONDS)
```

**Step 3:** Raise explicit error instead of silently returning None (lines 60–63)

```python
# BEFORE
if "Note" in data:
    logger.warning(f"API rate limited: {data['Note']}")
    return None

# AFTER
if "Note" in data:
    raise APIRateLimitError(f"API rate limited or quota exceeded: {data['Note']}")
if "Error Message" in data:
    raise APIError(f"API error: {data['Error Message']}")
```

**Step 4:** Define custom exceptions at the top of `fetcher.py` (after imports, before functions)

```python
class APIError(Exception):
    """Base exception for API-related errors."""
    pass

class APIRateLimitError(APIError):
    """Raised when API rate limit or quota is exceeded."""
    pass
```

**Result:** Now API failures raise exceptions instead of being silently swallowed, making them impossible to miss.

---

#### Option 3: Use a Real API Key & Service

**File:** `config/settings.py`

```python
# BEFORE
FETCH_REAL_DATA: bool = True  # Uses demo key (rate-limited)
API_TIMEOUT_SECONDS: int = 10

# AFTER
FETCH_REAL_DATA: bool = True
API_TIMEOUT_SECONDS: int = 10
# Add your real API key from Alpha Vantage or use an alternative service
ALPHA_VANTAGE_API_KEY: str = "your_real_api_key_here"
```

Then update `data/fetcher.py` line 14:

```python
# BEFORE
ALPHA_VANTAGE_API_KEY = "demo"

# AFTER (import from settings)
from config.settings import ALPHA_VANTAGE_API_KEY
```

**Result:** Real API data is fetched successfully (assuming valid key and network connectivity).

---

---

## Bug 1: KeyError — 'Close' Column Not Found

### When It Appears

**Crash Output:**

```
======================================================================
[STAGE 2/6] Feature Engineering — START
Attempting to fetch real stock data from Alpha Vantage API...
API rate limited: Thank you for using Alpha Vantage API...
API data fetch failed or timed out — falling back to sample data.
Data loaded: 500 rows | columns: ['Date', 'Open', 'High', 'Low', 'close_price', 'Volume']

======================================================================
PIPELINE FAILED — full traceback below:

Traceback (most recent call last):
  File "c:\...\stock_forecaster\main.py", line 100, in run_pipeline
    lag_df, rolling_df, target = build_feature_matrix(df)
  File "c:\...\stock_forecaster\features\feature_pipeline.py", line 29, in build_feature_matrix
    target = df["Close"].shift(-1).rename("target")
KeyError: 'Close'
```

---

### Error Details

| Field | Value |
|-------|-------|
| **File** | `features/feature_pipeline.py` (line 29) |
| **Root Cause** | Column is renamed to `'close_price'` in `loader.py` (line 55), but `feature_pipeline.py` still references the original name `'Close'` |
| **Impact** | Pipeline crashes at Stage 2, before model training begins |
| **Severity** | High (immediate crash) |

---

### How to Fix It

#### Solution: Use the Correct Column Name

**File:** `features/feature_pipeline.py`

**Change line 29:**

```python
# BEFORE
target = df["Close"].shift(-1).rename("target")

# AFTER
target = df["close_price"].shift(-1).rename("target")
```

**Explanation:**
- The loader renames `"Close"` → `"close_price"` for internal consistency
- Feature pipeline must use the renamed column name
- This aligns the column name with `TARGET_COLUMN` from config

**Verify the fix:**
- The log message on line 57 shows the loaded data has column `'close_price'` (not `'Close'`)
- When you run the pipeline again, it should proceed past Stage 2

---

---

## Bug 2: ValueError — Shape Mismatch in Array Concatenation

### When It Appears

**Crash Output:**

```
======================================================================
[STAGE 2/6] Feature Engineering — COMPLETE
[STAGE 3/6] Model Training — START
Assembling feature matrix — lag: (490, 5), rolling: (500, 12)
Training on X: ..., y: ...

======================================================================
PIPELINE FAILED — full traceback below:

Traceback (most recent call last):
  File "c:\...\stock_forecaster\main.py", line 100, in run_pipeline
    model, X, y = train(lag_df, rolling_df, target)
  File "c:\...\stock_forecaster\models\trainer.py", line 38, in train
    X_array = np.concatenate([lag_df.values, rolling_df.values], axis=1)
ValueError: all the input array dimensions for the concatenation axis 0 must match exactly, 
but along dimension 0, the array at index 0 has size 490 and the array at index 1 has size 500
```

---

### Error Details

| Field | Value |
|-------|-------|
| **File** | `models/trainer.py` (line 38) |
| **Root Cause** | `lag_features.py` drops NaN rows (→ 490 rows), but `rolling_features.py` retains them (→ 500 rows). When concatenating, row counts must match. |
| **Impact** | Pipeline crashes at Stage 3, during model training |
| **Severity** | High (immediate crash) |

---

### How to Fix It

#### Solution: Align NaN Handling Between Lag & Rolling Features

**Choice A: Drop NaN in rolling features (matches lag behavior)**

**File:** `features/rolling_features.py`

**Add this after line 31 (after the rolling features are computed):**

```python
# BEFORE (end of function)
logger.info(
    f"Rolling features computed — {len(roll_df)} rows (NaN rows retained) | "
    f"columns: {list(roll_df.columns)}"
)
return roll_df

# AFTER
before = len(roll_df)
roll_df = roll_df.dropna().reset_index(drop=True)
logger.info(
    f"Rolling features computed — retained {len(roll_df)} rows "
    f"({before - len(roll_df)} NaN rows dropped) | "
    f"columns: {list(roll_df.columns)}"
)
return roll_df
```

**Result:** Both `lag_df` and `rolling_df` now have 490 rows (NaN dropped), so concatenation succeeds.

---

#### OR Choice B: Don't drop NaN in lag features (matches rolling behavior)

**File:** `features/lag_features.py`

**Remove the dropna() call (lines 25-26):**

```python
# BEFORE
before = len(lag_df)
lag_df = lag_df.dropna().reset_index(drop=True)

logger.info(
    f"Lag features computed — retained {len(lag_df)} rows "
    f"({before - len(lag_df)} NaN rows dropped) | "
    f"columns: {list(lag_df.columns)}"
)
return lag_df

# AFTER
logger.info(
    f"Lag features computed — {len(lag_df)} rows (NaN rows retained) | "
    f"columns: {list(lag_df.columns)}"
)
return lag_df
```

Then in `models/trainer.py` (line 44-46), handle NaN rows when masking:

```python
# This code already exists and handles it:
mask = ~X.isnull().any(axis=1)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
```

**Result:** Both DataFrames have 500 rows, concatenation succeeds, but NaN rows are dropped during training anyway.

---

**Recommendation:** Use **Choice A** (drop NaN in rolling features) because:
- It's consistent with lag feature behavior
- It keeps the dataset smaller earlier (better memory efficiency)
- The log messages clearly show how many rows were dropped

---

---

## Bug 3: IndexError — Empty Fold in Backtesting

### When It Appears

**Crash Output:**

```
======================================================================
[STAGE 3/6] Model Training — COMPLETE
[STAGE 4/6] Backtesting — START
Backtesting — dataset: 490 rows | initial_train: 252 | step: 50 | folds: 6
Fold 00 — train rows: 252 | test rows: 50 | test range: [252:302]
Fold 01 — train rows: 302 | test rows: 50 | test range: [302:352]
Fold 02 — train rows: 352 | test rows: 50 | test range: [352:402]
Fold 03 — train rows: 402 | test rows: 50 | test range: [402:452]
Fold 04 — train rows: 452 | test rows: 50 | test range: [452:502]
Fold 05 — train rows: 502 | test rows: 0 | test range: [502:552]

======================================================================
PIPELINE FAILED — full traceback below:

Traceback (most recent call last):
  File "c:\...\stock_forecaster\main.py", line 100, in run_pipeline
    backtest_results = run_backtest(model, X, y)
  File "c:\...\stock_forecaster\evaluation\backtester.py", line 64, in run_backtest
    fold_metrics = compute_fold_metrics(y_test.values, y_pred, fold)
  File "c:\...\stock_forecaster\evaluation\metrics.py", line 22, in compute_fold_metrics
    baseline_price = y_true[0]
IndexError: index 0 is out of bounds for axis 0 with size 0
```

---

### Error Details

| Field | Value |
|-------|-------|
| **File** | `evaluation/metrics.py` (line 22) |
| **Root Cause** | Hardcoded backtest config (initial_train=252, step=50, n_splits=6) overshoots dataset length (~490 rows). Last fold has train_end=502, which exceeds available data, leaving test fold empty. |
| **Impact** | Pipeline crashes at Stage 4, during backtesting |
| **Severity** | High (immediate crash) |

---

### How to Fix It

#### Solution: Validate & Adjust Backtest Configuration

**File:** `config/settings.py`

**Problem Analysis:**
- Dataset: ~490 rows
- Config: initial_train_size=252, step_size=50, n_splits=6
- Calculation: 252 + (5 × 50) = 502 rows needed, but only 490 available
- Result: Last fold (fold 5) tries to test on rows [502:552], which is empty

**Fix Option A: Reduce number of folds**

```python
# BEFORE
BACKTEST_CONFIG: dict = {
    "initial_train_size": 252,
    "step_size": 50,
    "n_splits": 6,
}

# AFTER
BACKTEST_CONFIG: dict = {
    "initial_train_size": 252,
    "step_size": 50,
    "n_splits": 5,  # Reduced from 6
}
```

**Calculation:** 252 + (4 × 50) = 452 rows → Fold 4 tests on [452:502], which is valid ✓

---

#### OR Fix Option B: Reduce step size

```python
# BEFORE
BACKTEST_CONFIG: dict = {
    "initial_train_size": 252,
    "step_size": 50,
    "n_splits": 6,
}

# AFTER
BACKTEST_CONFIG: dict = {
    "initial_train_size": 252,
    "step_size": 40,  # Reduced from 50
    "n_splits": 6,
}
```

**Calculation:** 252 + (5 × 40) = 452 rows → Fold 5 tests on [452:492], which is valid ✓

---

#### OR Fix Option C: Validate at runtime (in backtester.py)

**File:** `evaluation/backtester.py`

Add validation at the beginning of `run_backtest()` (after line 32):

```python
# Add after line 32, before the loop
required_rows = initial_train + (n_splits - 1) * step
if required_rows > n:
    logger.warning(
        f"Backtest config requires {required_rows} rows but only {n} available. "
        f"Reducing n_splits from {n_splits} to {(n - initial_train) // step + 1}"
    )
    n_splits = (n - initial_train) // step + 1
```

**Result:** Dynamically adjusts `n_splits` to fit the data size.

---

**Recommendation:** Use **Fix Option A** (reduce n_splits to 5) because:
- It's simple and explicit
- The config clearly states what will happen
- Backtesting still has sufficient folds for validation
- No runtime branching logic needed

---

---

## Bug 4: AttributeError — 'NoneType' Object Has No Attribute 'predict'

### When It Appears

**Crash Output:**

```
======================================================================
[STAGE 4/6] Backtesting — COMPLETE
[STAGE 5/6] Prediction — START
Loading model from registry for inference on 20 samples.

======================================================================
PIPELINE FAILED — full traceback below:

Traceback (most recent call last):
  File "c:\...\stock_forecaster\main.py", line 100, in run_pipeline
    predictions = generate_predictions(X_latest)
  File "c:\...\stock_forecaster\models\predictor.py", line 22, in generate_predictions
    predictions = model.predict(X)
AttributeError: 'NoneType' object has no attribute 'predict'
```

---

### Error Details

| Field | Value |
|-------|-------|
| **File** | `models/predictor.py` (line 22) |
| **Root Cause** | `register_model()` stores the model under key `"model"`, but `get_model()` retrieves key `"estimator"`. Key mismatch causes `get_model()` to return None. |
| **Impact** | Pipeline crashes at Stage 5, during prediction |
| **Severity** | High (immediate crash) |

---

### How to Fix It

#### Solution: Fix the Key Mismatch in Model Registry

**File:** `models/model_registry.py`

**Option A: Change the storage key to match retrieval (line 20)**

```python
# BEFORE (line 20)
_registry["model"] = model

# AFTER
_registry["estimator"] = model
```

**Result:** Both store and retrieve use `"estimator"` key → Consistent ✓

---

#### OR Option B: Change the retrieval key to match storage (line 40)

```python
# BEFORE (line 40)
return _registry.get("estimator")

# AFTER
return _registry.get("model")
```

**Result:** Both store and retrieve use `"model"` key → Consistent ✓

---

**Recommendation:** Use **Option A** (change storage to `"estimator"`) because:
- The parameter name in `get_model(name: str = "forecaster")` suggests the key should be semantic
- Convention in ML frameworks often uses `"model"` or `"estimator"` for the primary model
- Consistent with sklearn terminology (`RandomForestRegressor` is often called an "estimator")

---

---

## Bug 5: FileNotFoundError — Cache Directory Not Created

### When It Appears

**Crash Output:**

```
======================================================================
[STAGE 5/6] Prediction — COMPLETE
[STAGE 6/6] Report Generation — START
Loading backtest results from cache for report generation.

======================================================================
PIPELINE FAILED — full traceback below:

Traceback (most recent call last):
  File "c:\...\stock_forecaster\main.py", line 100, in run_pipeline
    generate_report(predictions)
  File "c:\...\stock_forecaster\evaluation\report.py", line 27, in generate_report
    results = load_results()
  File "c:\...\stock_forecaster\utils\cache.py", line 36, in load_results
    with open(CACHE_RESULTS_PATH, "rb") as fh:
FileNotFoundError: [Errno 2] No such file or directory: 
'c:\...\stock_forecaster\cache\results\backtest_results.pkl'
```

---

### Error Details

| Field | Value |
|-------|-------|
| **File** | `utils/cache.py` (line 36) |
| **Root Cause** | `save_results()` attempts to write to `CACHE_RESULTS_PATH`, but the parent directory `CACHE_DIR` was never created. The FileNotFoundError from `open()` is silently caught (line 21), so the caller thinks the file was saved. Later, `load_results()` tries to open the non-existent file and crashes. |
| **Impact** | Pipeline crashes at Stage 6, during report generation |
| **Severity** | High (immediate crash) |

---

### How to Fix It

#### Solution A: Create the Cache Directory (Best Fix)

**File:** `utils/cache.py`

**Modify the `save_results()` function (lines 18-24):**

```python
# BEFORE
def save_results(results: list) -> None:
    """..."""
    try:
        with open(CACHE_RESULTS_PATH, "wb") as fh:
            pickle.dump(results, fh)
    except FileNotFoundError:
        pass
    logger.info(f"Backtest results persisted to {CACHE_RESULTS_PATH}")

# AFTER
def save_results(results: list) -> None:
    """..."""
    # Create parent directory if it doesn't exist
    CACHE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(CACHE_RESULTS_PATH, "wb") as fh:
            pickle.dump(results, fh)
        logger.info(f"Backtest results persisted to {CACHE_RESULTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise
```

**Key changes:**
1. Create the directory before attempting to write: `CACHE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)`
2. Don't silently catch exceptions → Re-raise them so the caller knows the save failed
3. Move the success log inside the try block (only log if save actually succeeded)

---

#### OR Solution B: Create Cache Directory in Settings

**File:** `config/settings.py`

Add this after line 9:

```python
# Create cache directory at module import time
CACHE_DIR.mkdir(parents=True, exist_ok=True)
```

**Result:** Directory is created when settings are imported, before `save_results()` is called.

---

#### OR Solution C: Fix at Initialization (main.py)

**File:** `stock_forecaster/main.py`

Add this at the top of the `run_pipeline()` function (after line 38):

```python
def run_pipeline() -> None:
    # Ensure cache directory exists
    from config.settings import CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(_DIVIDER)
    # ... rest of function
```

---

**Recommendation:** Use **Solution A** (create directory in save_results) because:
- The directory is only needed when saving
- Explicit is better than implicit
- Error handling re-raises the exception instead of silently swallowing it
- Follows the principle: failures should fail hard, not silently
- The logs clearly indicate success or failure

---

---

## Testing Checklist

After fixing each bug, use this checklist to verify:

### Bug 6 (Silent API Failure)
- [ ] Run pipeline with `FETCH_REAL_DATA=True` → See warning in logs about API failure
- [ ] Run pipeline with `FETCH_REAL_DATA=False` → No API attempt, uses sample data
- [ ] Verify logs clearly show whether API was attempted and what happened

### Bug 1 (KeyError: 'Close')
- [ ] Run pipeline → Should proceed past Stage 2
- [ ] Check Stage 2 log message shows target column created successfully
- [ ] Feature matrix is built without errors

### Bug 2 (ValueError: shape mismatch)
- [ ] Run pipeline → Should proceed past Stage 3
- [ ] Check logs show lag_df and rolling_df have same number of rows
- [ ] Model training completes with "Model training complete" log message

### Bug 3 (IndexError: empty fold)
- [ ] Run pipeline → Should proceed past Stage 4
- [ ] Check backtesting logs show all folds have non-zero test rows
- [ ] No folds with "test rows: 0"
- [ ] "Backtesting complete" message appears

### Bug 4 (AttributeError: None.predict)
- [ ] Run pipeline → Should proceed past Stage 5
- [ ] Check Stage 5 log shows predictions generated
- [ ] "Predictions generated" message appears with shape

### Bug 5 (FileNotFoundError: cache)
- [ ] Run pipeline → Should proceed past Stage 6 and complete successfully
- [ ] Check that `cache/results/backtest_results.pkl` is created
- [ ] Check that reports are generated in `reports/` directory
- [ ] "PIPELINE COMPLETE" message appears

---

## Quick Fix Reference

| Bug | Quick Fix |
|-----|-----------|
| **Bug 6** | Set `FETCH_REAL_DATA = False` in settings.py |
| **Bug 1** | Change `df["Close"]` → `df["close_price"]` in feature_pipeline.py:29 |
| **Bug 2** | Add `roll_df = roll_df.dropna().reset_index(drop=True)` in rolling_features.py |
| **Bug 3** | Change `n_splits: 6,` → `n_splits: 5,` in settings.py |
| **Bug 4** | Change `_registry["model"]` → `_registry["estimator"]` in model_registry.py:20 |
| **Bug 5** | Add `CACHE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)` in cache.py:save_results() |

---

## Error Message Patterns

Use these patterns to quickly identify which bug is occurring:

| Error Message | Bug # | File |
|---|---|---|
| `KeyError: 'Close'` | Bug 1 | feature_pipeline.py:29 |
| `ValueError: ... concatenation axis ...` | Bug 2 | trainer.py:38 |
| `IndexError: index 0 is out of bounds` | Bug 3 | metrics.py:22 |
| `AttributeError: 'NoneType' object has no attribute 'predict'` | Bug 4 | predictor.py:22 |
| `FileNotFoundError: ... backtest_results.pkl` | Bug 5 | cache.py:36 |
| `WARNING: API rate limited` (in logs, but no crash) | Bug 6 | fetcher.py / loader.py |

---

## Version History

- **v1.1** — Added Bug 6 (Silent API Failure)
- **v1.0** — Initial documentation for Bugs 1–5
