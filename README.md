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

<!--
Yes, we do use the WHO region mapping — but only for labeling and results grouping, not for actual regional forecasting.
Here's exactly what happens in the code:

What we actually do with the WHO region data:
Mapping – Each country is assigned a WHO region (e.g., Germany → EURO).

Display – The region appears in:

The performance table (printed and saved to report)

The bar chart x‑axis labels

The final summary text

Example from the code (Step 4 or report_generator.py):

python
region_map = {
    'Germany': 'EURO',
    'United States': 'AMRO',
    'India': 'SEARO',
    'South Africa': 'AFRO',
    'Japan': 'WPRO'
}
Then later:

python
table_data.append([region, country, f"{arimam:.1f}%", f"{sarimaxm:.1f}%", better])
What we do NOT do (and could be added):
❌ Aggregate cases at the WHO region level (e.g., sum all countries in EURO and forecast that)

❌ Train a separate model per region (we train per country, not per region)

❌ Compare region‑wide seasonality across continents

So the region is used as a descriptive label, not as a forecasting unit.

- Why the code is written this way
The original report you shared mentions WHO regions, but the actual forecasting is done at the country level (Germany, India, etc.) and then presented with their region names. This is typical for a beginner/mid-level project: you show that you understand the regional context, but you don't overcomplicate the modeling.

- If you want true regional forecasting (aggregated data)
You could modify the preprocessing to:

python code :
# Group by WHO region and date, sum cases
regional_weekly = df.groupby(['who_region', 'date'])['new_cases'].sum().reset_index()

-->

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
