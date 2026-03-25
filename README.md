# MAI391_Project
# MAI391_Project: A Hybrid Weather Forecasting System for Ho Chi Minh City

## Project Overview
This repository contains the code and data pipeline for a hybrid weather forecasting framework designed specifically for the rapidly changing climate of Ho Chi Minh City[cite: 4, 12]. The system is built on historical meteorological data collected from the Open-Meteo Archive API[cite: 4, 14]. 

The project is divided into two main architectural branches:
* **Classification Branch (Implemented):** An XGBoost multi-class classifier designed to predict next-day weather condition categories based on chronological data splits[cite: 6, 27].
* **Regression Branch (Prototype):** A Physics-Informed Temporal Attention Network (PITAN) designed to estimate quantitative variables like temperature and humidity under physics-guided constraints[cite: 7, 28, 43].

## File Structure
The project is organized into four sequential Jupyter Notebooks to demonstrate the full machine learning pipeline from data acquisition to model evaluation[cite: 47]:

* **01_Data_Acquisition_and_Preprocessing.ipynb**: Handles the ingestion of historical weather data, aggregates hourly variables into daily statistics, maps WMO weather codes to categorical conditions, and applies Z-score normalization[cite: 5, 55, 60, 61, 65].
* **02_Exploratory_Data_Analysis.ipynb**: Provides visual mathematical analyses of the dataset, including boxplots of monthly temperature distributions, correlation heatmaps of meteorological features, and precipitation bar charts to identify seasonal patterns[cite: 118, 141, 195, 216].
* **03_Weather_Classification_XGBoost.ipynb**: Implements the multi-class XGBoost model for next-day weather state prediction, utilizing chronological train/test splitting to preserve temporal dependencies and avoid data leakage[cite: 67, 70].
* **04_PITAN_Regression_Prototype.ipynb**: Constructs the prototype of the PITAN architecture (1D-CNN + LSTM + Temporal Attention) and implements the custom physics-guided loss function to penalize physically inconsistent predictions[cite: 73, 77].

## Setup and Execution
1. Ensure you have the raw dataset (`weatherHCM.csv`) in the root directory.
2. Install the required dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, and `torch`.
3. Run the notebooks strictly in numerical order (01 to 04). Notebook 01 generates the `hcm_weather_preprocessed.csv` file, which is required by all subsequent notebooks.

## Key Findings
* **Model Performance:** The XGBoost classification pipeline achieved an overall accuracy of 44.75% for next-day weather condition prediction[cite: 8, 256].
* **Class Imbalance:** Dominant classes like Overcast and Moderate rain were easier to predict, while minority classes like Dense drizzle struggled due to class imbalance[cite: 259, 260, 326].
* **Feature Importance:** Mathematical feature importance analysis revealed that pressure, humidity, cloud cover, and wind speed are the most influential predictors for local weather states[cite: 261, 327].
