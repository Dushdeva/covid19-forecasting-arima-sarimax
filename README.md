# 🦠 COVID-19 Case Forecasting — ARIMA & SARIMAX

> **Time Series Forecasting · Multi-Country Analysis · WHO Regional Comparison**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dushdeva/covid19-forecasting-arima-sarimax/blob/main/COVID_Forecasting_Project.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-orange)
![pmdarima](https://img.shields.io/badge/pmdarima-auto--arima-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

Built an **end-to-end time series forecasting pipeline** to predict weekly COVID-19 cases across **5 WHO regions** using classical statistical models. The project covers the full ML workflow — data ingestion, preprocessing, automated hyperparameter tuning, model evaluation, and structured reporting.

**Core objective:** Compare ARIMA and SARIMAX forecasting performance across geographically and epidemiologically diverse regions, and surface actionable model selection recommendations per region.

---

## 📊 Results Summary

| Country | WHO Region | ARIMA MAPE | SARIMAX MAPE | Better Model |
|---------|-----------|------------|--------------|-------------|
| Germany | EURO | 0.00% | 2,891% | ARIMA ✅ |
| United States | AMRO | 0.00% | 442,238% | ARIMA ✅ |
| India | SEARO | 868,355% | 21,149% | SARIMAX ✅ |
| South Africa | AFRO | 29,939% | 331% | SARIMAX ✅ |
| Japan | WPRO | 429,400% | 46,331% | SARIMAX ✅ |

> ⚠️ **Note on high MAPE values:** MAPE breaks down when actual case counts pass through near-zero values (end-of-wave troughs), causing division-by-near-zero inflation. This is a known limitation of MAPE as a metric for COVID-19 data which has extreme spikes and prolonged zero-adjacent periods. **Relative comparison between models per country remains valid.** Identified fix: switch to SMAPE or MAE for future iterations — tracked in improvements below.

**Key finding:** SARIMAX outperforms ARIMA in 4 out of 5 countries. Germany and USA show near-perfect ARIMA fit on the training window but SARIMAX degrades — likely due to the forecast window landing on a wave trough. India, South Africa, and Japan all show SARIMAX as the stronger model.

---

## 🏗️ Pipeline Architecture

```
OWID Dataset (live URL)
        │
        ▼
  Data Ingestion & Cleaning
  (pd.read_csv → date parsing → country filter)
        │
        ▼
  Weekly Aggregation
  (resample('W').sum() per country)
        │
        ▼
  Train / Test Split (last 8 weeks held out)
        │
      ┌─┴─────────────────┐
      ▼                   ▼
  auto_arima()        SARIMAX(1,1,1)
  (AIC minimization)  (seasonal_order=(1,0,1,52))
      │                   │
      └──────┬────────────┘
             ▼
       MAPE Evaluation
             │
             ▼
   Visualizations + Per-Country Report
```

---

## 🔧 Technical Details

**Models:**
- **ARIMA** — order (p,d,q) selected automatically via `pmdarima.auto_arima` using AIC minimization
- **SARIMAX** — fixed order (1,1,1) with seasonal component (1,0,1,52) capturing annual weekly patterns

**Evaluation Metric:** MAPE (Mean Absolute Percentage Error) — chosen for interpretability across countries with very different case magnitudes

**Data Source:** [Our World in Data (OWID)](https://github.com/owid/covid-19-data) — live pull on every notebook run, ensuring reproducibility with latest data

**Forecast Horizon:** 8 weeks ahead (held-out test set)

---

## 📁 Repository Structure

```
covid19-forecasting-arima-sarimax/
│
├── COVID_Forecasting_Project.ipynb   # Main notebook (Colab-ready)
├── requirements.txt                  # Dependencies for local use
├── README.md
│
├── images/                           # Auto-generated on run
│   ├── raw_ts.png                    # Raw time series for all countries
│   ├── model_comparison.png          # ARIMA vs SARIMAX MAPE bar chart
│   └── india_forecast.png            # India forecast overlay
│
└── reports/
    └── summary.txt                   # Per-country model recommendation report
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Click the **Open in Colab** badge above
2. `Runtime → Run all`
3. The notebook will automatically:
   - Download live COVID data from OWID
   - Train both models for all 5 countries
   - Generate comparison charts
   - Print per-country recommendations
   - Offer a ZIP download of all outputs

**No setup needed. Runs on any device with a browser.**

### Option 2: Local (Python 3.10+)

```bash
git clone https://github.com/Dushdeva/covid19-forecasting-arima-sarimax.git
cd covid19-forecasting-arima-sarimax
pip install -r requirements.txt
jupyter notebook COVID_Forecasting_Project.ipynb
```

---

## 📈 Sample Output

| Model Comparison | India Forecast |
|:---:|:---:|
| ![Model Comparison](https://github.com/Dushdeva/covid19-forecasting-arima-sarimax/blob/main/Report/images/model_comparison.png?raw=true) | ![India Forecast](https://github.com/Dushdeva/covid19-forecasting-arima-sarimax/blob/main/Report/images/india_forecast.png?raw=true) |

**Raw Time Series (all 5 countries):**

![Raw Time Series](https://github.com/Dushdeva/covid19-forecasting-arima-sarimax/blob/main/Report/images/raw_ts.png?raw=true)

---

## 🔮 Identified Improvements

- [ ] **Switch evaluation metric** from MAPE to SMAPE or MAE to handle near-zero actual values in COVID wave troughs
- [ ] Add **mobility data** (Google Community Mobility Reports) as exogenous variable in SARIMAX
- [ ] Implement **Prophet** model for comparison (handles multiple seasonality natively)
- [ ] Deploy interactive forecast dashboard using **Streamlit**
- [ ] Add **confidence intervals** to forecast visualization
- [ ] Improve South Africa forecasting with data imputation strategies

---

## 🛠️ Tech Stack

| Component | Library/Tool |
|-----------|-------------|
| Data manipulation | `pandas`, `numpy` |
| Time series models | `statsmodels` (ARIMA, SARIMAX) |
| Auto hyperparameter tuning | `pmdarima` (auto_arima) |
| Model evaluation | `scikit-learn` (MAPE) |
| Visualization | `matplotlib`, `seaborn` |
| Environment | Google Colab / Jupyter |

---

## 📄 Acknowledgments

- **Data:** [Our World in Data](https://ourworldindata.org/covid-cases)
- **Libraries:** statsmodels, pmdarima, scikit-learn, pandas

---

*Built by [Devang Yadav](https://github.com/Dushdeva) — B.Tech CSE (AI), SKIT Jaipur*
