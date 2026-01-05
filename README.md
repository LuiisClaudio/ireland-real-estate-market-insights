# 🇮🇪 Ireland Real Estate Market Insights Dashboard

## Dashboard Link: https://irish-real-estate-market.streamlit.app/

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

The dashboard is redefined into **6 narrative modules**, designed to guide the user from macro-trends to micro-insights:

### 1. ⏱️ Module A: Temporal Dynamics
*Focus: How has the market evolved over time?*
*   **Market Velocity KPI**: Immediate pulse of the market (Volume, Price, Units).
*   **Trend Analysis**: Tracking the median price recovery and supply-demand elasticity (Volume-Price Correlation).
*   **Seasonality & Heatmaps**: Visualizing the "Spring Bloom" and structural market breaks (e.g., COVID-19 lockdowns).
*   **Regional Divergence**: Contrasting Dublin's performance against top counties.

### 2. 🗺️ Module B: Geospatial Intelligence
*Focus: Where is the value concentrated?*
*   **National Overview**: East-West economic divide visualized via Scatter Maps and Provincial Treemaps.
*   **Premium Rankings**: Identifying the top 20 most expensive postcodes/areas.
*   **Hyper-Local Density**: 3D Hexagon layers and Mapbox scatters identifying urban "hotspots" and price clusters.

### 3. 💰 Module C: Distribution & Affordability
*Focus: How accessible is the market?*
*   **Affordability Tiers**: Breaking down inventory into 'Starter', 'Middle', and 'Premium' segments.
*   **Variance Analysis**: Box plots revealing price heterogeneity within counties.
*   **Comparative Stats**: "New Build Premium" (Violin plots) and "Bracket Creep" over time (Ridgeline plots).

### 4. 🏠 Module D: Attribute Correlations
*Focus: What drives property value?*
*   **Property Composition**: Sunburst charts exploring the hierarchy of Province -> County -> Type.
*   **Size & VAT**: Analyzing the impact of floor area and new supply (VAT status) on pricing.
*   **Multivariate Flows**: Parallel coordinate plots tracing common property profiles.

### 5. 🔮 Module E: Predictive Modeling
*Focus: Where is the market heading?*
*   **ARIMA vs. SARIMA**: Comparing standard trend models against **Seasonal ARIMA** to prove the predictability of market cycles.
*   **Forecast**: Generating future price confidence intervals to aid decision-making.

### 6. � Module F: Clustering Analysis
*Focus: Can we find hidden market segments?*
*   **K-Means (Hard Clustering)**: Mathematically defining distinct market tiers (e.g., Economy vs. Luxury) based on price and location.
*   **Fuzzy C-Means (Soft Clustering)**: Identifying "bridge" properties that don't fit perfectly into one box, revealing the nuance of edge cases.

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
