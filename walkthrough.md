# Project Reorganization Walkthrough

I have reorganized the project structure to be more scalable and maintainable.

## Changes Made

### Directory Structure
- **`src/`**: Contains source code (`Dashboard.py`, `train_models.py`).
- **`data/raw/`**: Contains original CSV files.
- **`data/processed/`**: Contains processed CSV files used by the dashboard.
- **`models/`**: Contains trained `.pkl` models.

### Code Updates
- **`src/train_models.py`**: Updated to load raw data from `../data/raw/` and save artifacts to `../models/` and `../data/processed/`. Used absolute paths for robustness.
- **`src/Dashboard.py`**: Updated to load models and data from `../models/` and `../data/processed/`. Used absolute paths for robustness.

## Verification Results

### Automated Tests
Ran `python src/train_models.py` successfully.

```
--- Resultados de la Evaluación del Modelo de CHURN ---
Precisión del Modelo (Accuracy): 0.9628

--- Resultados de la Evaluación del Modelo de LEAD SCORING ---
Precisión del Modelo (Accuracy): 0.7622

Modelos y DataFrames guardados (.pkl y .csv) para el dashboard.
```

### Manual Verification
You can now run the dashboard using:
```bash
streamlit run src/Dashboard.py
```
Or from the root directory:
```bash
streamlit run src/Dashboard.py
```
Both will work because the scripts now use paths relative to themselves.
