# COVID-19 Forecasting using ARIMA & SARIMAX

> Predict COVID-19 cases across 5 WHO regions using ARIMA and SARIMAX models.

## Project Overview

This project compares **ARIMA** and **SARIMAX** models for forecasting weekly confirmed COVID-19 cases.  
We analyze five countries, each representing a different WHO region:

| Country        | WHO Region |
|----------------|------------|
| Germany        | EURO       |
| United States  | AMRO       |
| India          | SEARO      |
| South Africa   | AFRO       |
| Japan          | WPRO       |

### Key Questions
- Can time series models reliably predict COVID-19 cases?
- How do forecasts differ across WHO regions?
- Which model (ARIMA or SARIMAX) works better for this data?

## Repository Structure
```text
covid19-forecasting-arima-sarimax/
│
├── COVID_Forecasting_Project.ipynb # Main notebook (Google Colab)
├── README.md # This file
├── requirements.txt # Python dependencies (local use)
│
├── data/ # (auto-generated) raw data
├── images/ # (auto-generated) charts
│ ├── model_comparison.png
│ └── india_forecast.png
│
└── reports/ # (auto-generated) summary report
└── summary.txt
```


## How to Run

### Option 1: Google Colab (Recommended – no setup)

1. Click the **"Open In Colab"** badge at the top of this README.
2. In Colab, go to **Runtime → Run all**.
3. Wait 2-3 minutes – the notebook will:
   - Download the latest COVID-19 data
   - Train ARIMA and SARIMAX models
   - Generate comparison charts
   - Display a final summary report
   - Offer a download link for all results (ZIP file)

No installation required – works on any device with a browser.

### Option 2: Run locally (Python)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/covid19-forecasting-arima-sarimax.git
cd covid19-forecasting-arima-sarimax

# Install dependencies
pip install -r requirements.txt

# Run the notebook (or convert to Python script)
jupyter notebook COVID_Forecasting_Project.ipynb
```
# Acknowledgments
Data: Our World in Data
Libraries: statsmodels, pmdarima, scikit-learn
