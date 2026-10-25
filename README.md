# AgriPredict AI — Crop Yield Prediction System

AgriPredict AI is an end-to-end crop yield estimation and agricultural decision-support system tailored to Indian agriculture. Powered by a state-aware Gradient Boosting regression pipeline trained on nearly two decades of district-level agricultural records, AgriPredict AI estimates expected crop yield (tonnes per hectare, t/ha) and total harvest production (tonnes) based on historical state agronomic context, crop choice, cropping season, rainfall, and temperature.

---

## Features

- **Accurate Yield & Production Prediction**: Generates calibrated yield estimates (t/ha) and total harvest output (tonnes) for a specified cultivation area.
- **Dynamic Agronomic Combinations**: Cascading dropdown menus restrict user selections to historically verified `(State, Crop, Season)` combinations across 30 Indian states and union territories.
- **Smart Weather Auto-Fill**: Automatically loads historically typical seasonal weather (mean rainfall in mm and temperature in °C) for any selected state and season.
- **Dual-Layer Input Validation**:
  - *Hard Limit Enforcement*: Blocks physiologically impossible or out-of-scale inputs (HTTP 400).
  - *Seen Range Advisory Warnings*: Alerts users if entered weather deviates from the state's historical observed range while still computing predictions (HTTP 200 with warnings).
- **Historical Seasonal Trends**: Visualizes historical yield baselines per season for a specific crop and state, with fallback to national averages.
- **Transparent Validation & Metrics**: Serves temporal holdout performance benchmarks directly through a dedicated API endpoint.

---

## Tech Stack

- **Backend**: Python 3.12, Flask 3.1.3, Flask-CORS 6.0.5
- **Machine Learning & Data Processing**: Scikit-Learn 1.6.1, NumPy 2.3.4, Pandas 2.2.3, Joblib 1.4.2, XGBoost 3.4.1
- **Data Acquisition & Visualization**: Requests 2.34.2, Matplotlib 3.10.0
- **Frontend**: Vanilla HTML5, CSS3 (Modern Dark Theme, Glassmorphism), Vanilla JavaScript (ES6+)
- **External Data APIs**: NASA POWER meteorological API

---

## Folder Structure

```text
Crop Yield Prediction/
├── assets/                     # Frontend media assets (hero background images, UI icons)
├── backend/                    # Python application package containing API, training, and evaluation scripts
├── data/                       # Agricultural and climate data root directory
│   ├── processed/              # Cleaned tabular crop yield datasets and weather cache files
│   └── raw/                    # Raw Ministry of Agriculture APY historical crop dataset from Kaggle
├── models/                     # Production model binary, metadata, and evaluation results
│   └── variants/               # Serialized pipelines for ablation variants A, B, C, and D
├── public/                     # Static production frontend files served directly by Flask
├── reports/                    # Generated charts, exploratory data analysis plots, and comparison reports
│   ├── eda/                    # Visualizations of feature distributions, correlation matrices, and seasonal patterns
│   └── model/                  # Model diagnostic figures (predicted vs. actual yield, relative MAE per crop)
├── app.py                      # Root convenience entry point script to run the Flask backend
├── backend/__init__.py         # Package initialization marker for backend module
├── backend/ablation.py         # Evaluates feature ablation variants (A, B, C, D) on a 1997-2012 / 2013-2015 temporal split
├── backend/app.py              # Flask REST API server exposing /predict, /options, /trends, /metrics, and /health
├── backend/eda.py              # Exploratory data analysis script generating statistical summaries and visualizations
├── backend/prepare_data.py     # Cleans raw APY dataset, engineers features, and fetches NASA POWER climate data
├── backend/robustness.py       # Rolling-window temporal validation study evaluating model stability across 3 time windows
├── backend/soil_map.py         # Heuristic state-to-dominant-soil-type mapping dictionary (no longer used by the final model)
├── backend/test_api.py         # Comprehensive automated test suite asserting endpoints, status codes, and input validation
├── backend/train_final.py      # Trains final production Variant C model on full dataset (1997-2015) and exports artifacts
├── backend/train_model.py      # Baseline and candidate model experimentation script
├── index.html                  # Frontend single-page application interface
├── script.js                   # Frontend application logic, form handling, API integration, and validation
├── style.css                   # Custom modern styling, responsive layouts, and typography
├── start_app.bat               # Windows batch script to launch the backend server and open the web app
├── requirements.txt            # Pinned Python package dependencies for production execution
├── requirements-dev.txt        # Development dependencies with minimum version constraints
├── .python-version             # Python version specification (3.12)
└── .gitignore                  # Git exclusion rules for large datasets, caches, and backups
```

*(Note: `backend/soil_map.py` is no longer used by the final model, which utilizes `State` directly as a native categorical feature.)*

---

## Setup

### Prerequisites
- Python 3.12 installed on your system.

### Installation
1. Clone or extract the repository to your local machine.
2. Open a terminal in the project root directory.
3. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Option 1: Automated Start (Windows)
Double-click `start_app.bat` or run it from the terminal:
```cmd
start_app.bat
```
This script launches the Flask backend server and automatically opens `http://127.0.0.1:5000` in your default browser.

### Option 2: Manual Start
1. Start the backend Flask server:
   ```bash
   python backend/app.py
   ```
   The API will listen at `http://127.0.0.1:5000`.

2. Serve the frontend:
   Open a second terminal in the project root directory and start Python's HTTP server:
   ```bash
   python -m http.server 8000
   ```

3. Open your browser and navigate to:
   ```text
   http://localhost:8000/index.html
   ```

---

## API Endpoints

### 1. `GET /health`
Returns the operational status of the service and active model name.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/health
```

**Example Response:**
```json
{
  "model": "HistGradientBoostingRegressor (State-Aware)",
  "status": "ok"
}
```

### 2. `GET /options`
Returns the list of available states, valid `(State -> Crop -> Seasons)` hierarchy, typical seasonal weather stats, and validation hard limits.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/options
```

### 3. `GET /trends`
Returns historical average yield per season for a specific crop and state, or falls back to national averages if state is omitted.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:5000/trends?crop=Wheat&state=Punjab"
```

**Example Response:**
```json
{
  "crop": "Wheat",
  "level": "state",
  "state": "Punjab",
  "trends": [
    {
      "avg_yield_tha": 4.41,
      "season": "Rabi"
    }
  ],
  "valid_seasons": ["Rabi"]
}
```

### 4. `GET /metrics`
Returns rolling-window temporal validation benchmarks and training metadata from `models/metadata.json`.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/metrics
```

### 5. `POST /predict`
Estimates crop yield (t/ha) and total production (tonnes) for the given input parameters.

**Example Request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"state": "Punjab", "crop": "Wheat", "season": "Rabi", "rainfall": 84.1, "temperature": 17.1, "area": 100.0}'
```

**Example Response:**
```json
{
  "model_name": "HistGradientBoostingRegressor (State-Aware)",
  "note": "Estimate based on historical district-level data; typical error is about 30% of a crop's average yield.",
  "predicted_yield": 4.12,
  "total_production": 412.0,
  "warnings": []
}
```

---

## Data Sources

1. **Crop Production in India (APY Dataset)**:
   - Sourced from Kaggle's *"Crop Production in India"* dataset, compiled originally from the Ministry of Agriculture and Farmers Welfare (Government of India) Area, Production, and Yield (APY) agricultural statistics across Indian districts from 1997 through 2015.
2. **NASA POWER Meteorological Data**:
   - Seasonal precipitation and 2-meter ambient temperatures obtained from the **NASA POWER** (Prediction of Worldwide Energy Resources) climatology API, aggregated by state coordinates and crop growing seasons.

---

## Model Summary & Validation Results

The production model is a **HistGradientBoostingRegressor (Variant C: State-Aware)** using a log1p target transformation (`np.log1p(Yield)`). It natively handles categorical features (`Crop`, `Season`, `State`) and models non-linear interactions with continuous climate drivers (`Rainfall_mm`, `Temperature_C`).

### Hyperparameters
- `max_iter`: 150
- `learning_rate`: 0.08
- `min_samples_leaf`: 40
- `max_depth`: None
- `random_state`: 42

### Feature Ablation Study (1997–2012 Train, 2013–2015 Temporal Holdout)
Evaluated on **137,995** training samples and **16,817** future holdout samples:

| Variant | Features Included | R² | R² (log) | MAE (t/ha) | RMSE (t/ha) | Macro Crop Rel MAE (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Variant A**: Current features | `Crop`, `Season`, `Rainfall_mm`, `Temperature_C`, `Soil_Type`, `Area_ha` | 0.8932 | 0.9011 | 0.8651 | 2.7322 | 31.35% |
| **Variant B**: A without Area_ha | `Crop`, `Season`, `Rainfall_mm`, `Temperature_C`, `Soil_Type` | 0.8942 | 0.896 | 0.8947 | 2.7191 | 32.11% |
| **Variant C**: B with State (Soil_Type removed) | `Crop`, `Season`, `Rainfall_mm`, `Temperature_C`, `State` | 0.8919 | 0.9123 | 0.8151 | 2.7489 | 28.53% |
| **Variant D**: Crop x Season mean baseline | `Crop`, `Season` | 0.8919 | 0.8691 | 0.9893 | 2.749 | 35.06% |

### Rolling-Window Temporal Robustness Study (1997–2015)
To ensure the model generalizes across changing climate regimes and prevents temporal leakage, performance was validated across three distinct rolling forward test periods:

| Time Window | Train Period | Test Period | Metric | Variant B (Soil Type, No Area) | Variant C (State, No Soil Type - Final) | Variant D (Crop × Season Baseline) |
|:---|:---|:---|:---|:---:|:---:|:---:|
| **2007-2009** | 1997-2006 (82,815 rows) | 2007-2009 (27,007 rows) | MAE<br>R² (log)<br>Rel. MAE | 0.8996<br>0.8441<br>36.86% | **0.8277**<br>**0.8672**<br>**34.25%** | 0.9747<br>0.822<br>48.67% |
| **2010-2012** | 1997-2009 (109,822 rows) | 2010-2012 (28,173 rows) | MAE<br>R² (log)<br>Rel. MAE | 0.809<br>0.8692<br>32.67% | **0.7518**<br>**0.8952**<br>**28.74%** | 0.8891<br>0.8428<br>36.01% |
| **2013-2015** | 1997-2012 (137,995 rows) | 2013-2015 (16,817 rows) | MAE<br>R² (log)<br>Rel. MAE | 0.8947<br>0.896<br>32.11% | **0.8151**<br>**0.9123**<br>**28.53%** | 0.9893<br>0.8691<br>35.06% |
| **Mean Across Windows** | 1997–2012 | 2007–2015 | MAE<br>R² (log)<br>Rel. MAE | 0.8678<br>0.8698<br>33.88% | **0.7982**<br>**0.8916**<br>**30.51%** | 0.951<br>0.8446<br>39.91% |

---

## Limitations

- **District-Level Aggregation**: Historical agricultural data is collected at the district level and aggregated into state-level options, which may mask intra-district microclimate variance.
- **State-Level Weather Representation**: Weather features represent state-level seasonal aggregates, which tend to underestimate localized rainfall in wet hilly regions or mountainous terrains.
- **Missing Agronomic Inputs**: The dataset does not contain records for irrigation type/coverage, fertilizer application (NPK), pest outbreaks, soil salinity, or seed varieties.
- **Historical Temporal Boundary**: Records conclude in the year 2015; trends do not capture post-2015 climate anomalies, recent seed genetics, or modernized agricultural practices.
- **Estimates, Not Forecasts**: Model outputs represent historical conditional estimates based on typical seasonal conditions, not real-time meteorological forecasts.
