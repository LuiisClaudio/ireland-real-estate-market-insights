# 🇮🇪 Ireland Real Estate Market Insights Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📊 Project Overview

This project is a comprehensive **Data Analytics & Machine Learning application** designed to uncover deep insights into the Irish Residential Property Market. 

Built with **Python** and **Streamlit**, it serves as an interactive portfolio piece demonstrating advanced capabilities in:
- **Exploratory Data Analysis (EDA)**: Uncovering trends, seasonality, and regional disparities.
- **Geospatial Intelligence**: Mapping price distributions and urban density.
- **Time Series Forecasting**: Predicting future market trends using **SARIMA** and **ARIMA** models.
- **Unsupervised Learning**: Segmenting the market using **K-Means** and **Fuzzy C-Means** clustering algorithms.

---

## 🚀 Key Modules & Analytical Features

The dashboard is structured into 6 strategic modules, each targeting specific analytical questions:

### 1. 📈 Temporal Dynamics
*   **Market Velocity KPI**: Real-time pulse of market activity.
*   **Seasonality Analysis**: Identifying the "Spring Bloom" and "Winter Lull" cycles in transaction volumes.
*   **Volume-Price Correlation**: Analyzing the supply-demand elasticity.

### 2. 🗺️ Geospatial Intelligence
*   **Hyper-Local Scatter Maps**: Visualizing price density at a granular level.
*   **National Choropleths**: highlighting the East-West economic divide.
*   **Urban H3 Hexagons**: 3D density visualization of transaction hotspots.

### 3. 💰 Distribution & Affordability
*   **Market Tier Analysis**: Breaking down inventory by price bands (Budget vs. Premium).
*   **New vs. Second-Hand**: Violin plots analyzing the "New Build Premium".

### 4. 🔗 Attribute Correlations
*   **Multivariate Parallel Coordinates**: Tracing flows between County, Size, and Price.
*   **Market Composition**: Sunburst charts for hierarchical data exploration.

### 5. 🔮 Predictive Modeling (Time Series)
*   **SARIMA vs ARIMA**: 
    *   Implemented **Seasonal ARIMA (SARIMA)** to account for the strong seasonal components in real estate data.
    *   Performed stationarity checks and differencing ($d=1$) to ensure robust model performance.
    *   **Metric**: Evaluated models using RMSE and MAE on a hold-out test set.

### 6. 🤖 Clustering Analysis (Machine Learning)
*   **K-Means Clustering**: Segmented properties into distinct classes (e.g., "Economy", "Mid-Market", "Premium") based on Price and Location.
*   **Fuzzy C-Means**: Analyzed edge cases where properties exhibit characteristics of multiple segments (soft clustering).

---

## 🛠️ Technical Stack

*   **Core Logic**: Python (Pandas, NumPy)
*   **Visualization**: Plotly Express, Plotly Graph Objects, PyDeck (Geospatial)
*   **Machine Learning**: Scikit-Learn (K-Means, PCA), Statsmodels (SARIMAX, ARIMA)
*   **Dashboard Framework**: Streamlit
*   **Data Processing**: Data cleaning pipelines, outlier handling (IQR method), and feature engineering.

---

## 💻 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone https://github.com/LuiisClaudio/ireland-real-estate-market-insights.git
    cd ireland-real-estate-market-insights
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 🧠 Analyst's Note

This project was developed to demonstrate a "Full Stack" data science approach—moving beyond simple charts to actionable insights and predictive modeling. 

**Key Challenges Solved:**
*   **Data Quality**: Handled missing values and inconsistent county naming conventions in the raw Property Price Register dataset.
*   **Seasonality**: The raw data showed strong seasonal variance; implementing SARIMA significantly outperformed standard ARIMA models by capturing these yearly cycles.
*   **User Experience**: Designed the Streamlit interface to be intuitive for non-technical stakeholders while providing deep-dive capabilities for analysts.
