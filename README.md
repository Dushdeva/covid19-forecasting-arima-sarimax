# COVID-19 Forecasting Using ARIMA and SARIMAX

## Project Overview
This project focuses on forecasting COVID-19 cases using Time Series Analysis techniques.

The objective was to compare `ARIMA` and `SARIMAX` models across different WHO regions and evaluate their forecasting performance using error metrics.

## Dataset
In the `Dataset` folder

The dataset contains:
- Country/Region
- Date
- Confirmed Cases
- Deaths
- Recovered Cases

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Python-Docx
- Google Colab

## Methodology
1. Data Cleaning
2. Weekly Aggregation
3. Exploratory Data Analysis
4. ARIMA Model Development
5. SARIMAX Model Development
6. Performance Comparison
7. Forecast Visualization
8. Report Generation

## Results
- SARIMAX performed better in regions with strong seasonal patterns.
- ARIMA showed competitive performance in some regions.
- Forecast confidence intervals widened appropriately over time.

## Key Learnings
- Time series forecasting requires careful preprocessing.
- Seasonal patterns significantly impact model accuracy.
- Visual analysis is as important as numerical metrics.


# HOW TO RUN THE HEART DISEASE PREDICTION PROJECT

1. Make sure you have Python installed (version 3.7 or higher).

2. Open a terminal/command prompt inside the project folder
   (the folder that contains `main.py`, `requirements.txt`, and the dataset/ folder).

3. Install the required libraries:
   `pip install -r requirements.txt`

4. Verify that the file `dataset/heart.csv` exists.
   (If not, place your heart.csv inside the dataset/ folder.)

5. Run the script:
   `python main.py`

6. After execution, open the `Report` folder. You will find:
   - `report.html`   (the main report – open with any browser)
   - All graphs (PNG files)
   - `model_comparison.csv` and `data_summary.txt`

7. Double-click on `report.html` to see all visualizations and model results.

# Author

## Devang Yadav
