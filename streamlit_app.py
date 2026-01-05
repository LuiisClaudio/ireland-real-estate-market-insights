import streamlit as st
import pandas as pd
import plotly.express as px
import visualization_code as vc
import machine_learning_module as mlm
import sarima_model as se
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Valid Streamlit page configuration
st.set_page_config(page_title="Irish Real Estate Market Dashboard", layout="wide")

# Load and Cache Data
@st.cache_data
def load_data():
    # Load the cleaned dataset
    try:
        df = pd.read_csv('cleaned_real_estate_data.csv', low_memory=False)
    except FileNotFoundError:
        # Fallback to loading and cleaning raw if cleaned doesn't exist (though it does)
        # For now, assume cleaned exists based on file listing
        st.error("Data file not found.")
        return pd.DataFrame()

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date_of_Sale'])
    
    # Ensure Sale_Year/Month exist
    if 'Sale_Year' not in df.columns:
        df['Sale_Year'] = df['Date'].dt.year
    if 'Sale_Month' not in df.columns:
        df['Sale_Month'] = df['Date'].dt.month
        
    return df

@st.cache_data
def run_clustering_models(df):
    """Wrapper for clustering module"""
    return mlm.run_clustering_models(df)

@st.cache_data
def run_forecasting_models(df):
    """
    Fits SARIMA and ARIMA models and returns the forecasts.
    Cached to avoid re-training on every interaction.
    Wrapper around the logic moved to sarima_evaluation.py
    """
    return se.run_forecasting_models(df)

df_raw = load_data()

# Sidebar Filters
st.sidebar.title("Filters")

if not df_raw.empty:
    # Year Filter
    min_year = int(df_raw['Sale_Year'].min())
    max_year = int(df_raw['Sale_Year'].max())
    selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    # County Filter
    all_counties = sorted(df_raw['County'].unique().astype(str))
    selected_counties = st.sidebar.multiselect("Select Counties", all_counties )#, default=all_counties[:3] if len(all_counties) > 3 else all_counties)
    
    # Description Filter
    # Map 1/0 back to text for display if needed, or just use as is. 
    # CSV has 1/0. Let's provide a friendly filter.
    prop_type_options = {"All": None, "New": 1, "Second Hand": 0}
    selected_prop_type_label = st.sidebar.selectbox("Property Type", list(prop_type_options.keys()))
    selected_prop_type_val = prop_type_options[selected_prop_type_label]

    # Price Filter
    Q_inf = df_raw['Price'].quantile(0.05)
    Q_sup = df_raw['Price'].quantile(0.99)
    IQR = Q_sup - Q_inf

    lower_bound = int(Q_inf)
    upper_bound = int(Q_sup + 1.5 * IQR)

    min_price = lower_bound
    max_price = upper_bound
    selected_price_range = st.sidebar.slider("Select Price Range (€)", min_price, max_price, (min_price, max_price))

    # Apply Filters
    df = df_raw.copy()
    df = df[(df['Sale_Year'] >= selected_years[0]) & (df['Sale_Year'] <= selected_years[1])]
    df = df[(df['Price'] >= selected_price_range[0]) & (df['Price'] <= selected_price_range[1])]
    
    if selected_counties:
        df = df[df['County'].isin(selected_counties)]
        
    if selected_prop_type_val is not None:
        df = df[df['Description_of_Property'] == selected_prop_type_val]
else:
    st.warning("No data loaded.")
    df = pd.DataFrame()

# Navigation Structure - Groups of Visualizations
MODULES = {
    "Module A: Temporal Dynamics": "Analysis of market trends, velocity, and seasonal patterns over time.",
    "Module B: Geospatial Intelligence": "Geographic distribution of prices, volume, and density across Ireland.",
    "Module C: Distribution & Affordability": "Deep dive into price segments, affordability tiers, and property variations.",
    "Module D: Attribute Correlations": "Exploring relationships between property features like size, type, and VAT status.",
    "Module E: Predictive Modeling": "Forecasting future market trends using SARIMA and ARIMA models.",
    "Module F: Clustering Analysis": "Advanced market segmentation using K-Means and Fuzzy C-Means algorithms."
}

# Sidebar Navigation
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
selected_module = st.sidebar.radio("Select Module", list(MODULES.keys()))

# Main Dashboard
st.title("🇮🇪 Irish Real Estate Market Insights")
st.markdown(f"## **{selected_module}**") 
st.markdown(f"*{MODULES[selected_module]}*")
st.markdown("---")

# Display Info Helper
def display_story_segment(title, logic, insight):
    with st.expander(f"ℹ️ {title} - Insights", expanded=False):
        st.markdown(f"**Logic:** {logic}")
        st.markdown(f"**Insight:** {insight}")

if df.empty:
    st.warning("Please adjust filters to view data.")
else:
    # --- MODULE A: TEMPORAL DYNAMICS ---
    if selected_module == "Module A: Temporal Dynamics":
        st.markdown("### ⏱️ Market Velocity & Trends")
        
        # Row 1: KPI
        st.markdown("#### Market Velocity")
        display_story_segment("V1: Market Velocity KPI", "Dynamic sums/medians based on selection.", "Immediate pulse 💓 of the market.")
        vc.plot_market_velocity_kpi(df)
        st.markdown("---")

        # Row 2: Trends
        st.markdown("#### Price & Volume Trends")
        col1, col2 = st.columns(2)
        with col1:
             display_story_segment("V2: Median Price Trend", "Median Price filtered by Month/Year.", "Macro-recovery trajectory 📈.")
             st.plotly_chart(vc.plot_median_price_trend(df), use_container_width=True)
        with col2:
             display_story_segment("V6: Volume-Price Correlation", "Median Price vs Volume.", "Supply 📦 vs Price 💰 dynamics.")
             st.plotly_chart(vc.plot_volume_price_correlation(df), use_container_width=True)

        # Row 3: Seasonality & Heatmap
        st.markdown("#### Seasonal Patterns")
        col3, col4 = st.columns(2)
        with col3:
            display_story_segment("V4: Seasonality", "Transactions by Month.", "Spring Bloom 🌸 vs Winter Lull ❄️.")
            st.plotly_chart(vc.plot_seasonality(df), use_container_width=True)
        with col4:
            display_story_segment("V5: Market Heatmap", "2D Matrix of Activity.", "Structural breaks like COVID-19 🧱.")
            st.plotly_chart(vc.plot_market_heatmap(df), use_container_width=True)

        # Row 4: Divergence
        st.markdown("#### Regional Performance")
        display_story_segment("V3: Regional Divergence", "Top 5 Counties comparison.", "Decoupling of Dublin market 🏙️.")
        st.plotly_chart(vc.plot_regional_divergence(df), use_container_width=True)

    # --- MODULE B: GEOSPATIAL INTELLIGENCE ---
    elif selected_module == "Module B: Geospatial Intelligence":
        st.markdown("### 🗺️ Geographic Market Analysis")

        # Row 1: High Level Maps
        st.markdown("#### National Overview")
        col1, col2 = st.columns(2)
        with col1:
            display_story_segment("V8: National Price Scatter", "Mean Price by County.", "East-West Divide 🗺️.")
            fig_v8 = vc.plot_national_price_choropleth(df)
            if fig_v8: st.plotly_chart(fig_v8, use_container_width=True)
            else: st.warning("Geo data missing.")
        with col2:
            display_story_segment("V11: Provincial Treemap", "Hierarchy: Province -> County.", "Relative market weights ⚖️.")
            st.plotly_chart(vc.plot_provincial_treemap(df), use_container_width=True)

        # Row 2: Rankings
        st.markdown("#### Premium Areas")
        display_story_segment("V7: Premium Postcode Ranking", "Top 20 Areas by Price.", "Identifies 'Premium' tiers 💎.")
        st.plotly_chart(vc.plot_premium_postcode_ranking(df), use_container_width=True)

        # Row 3: Detailed Maps
        st.markdown("#### Hyper-Local Density")
        st.info("Note: These Visualizations require Latitude/Longitude data.")
        col3, col4 = st.columns(2)
        with col3:
             display_story_segment("V9: Hyper-Local Scatter", "Scatter on Mapbox.", "Local density 📍 and prices.")
             fig_v9 = vc.plot_hyper_local_scatter(df)
             if fig_v9: st.plotly_chart(fig_v9, use_container_width=True)
             else: st.warning("Latitude/Longitude data missing.")
        with col4:
             display_story_segment("V10: Urban Density Hexagon", "3D Hexagon Layer.", "Hotspots 🔥 of activity.")
             deck_v10 = vc.plot_urban_density_hexagon(df)
             if deck_v10: st.pydeck_chart(deck_v10)
             else: st.warning("Latitude/Longitude data missing.")

    # --- MODULE C: DISTRIBUTION & AFFORDABILITY ---
    elif selected_module == "Module C: Distribution & Affordability":
        st.markdown("### 💰 Price Distribution & Affordability")

        # Row 1: Distribution
        col1, col2 = st.columns(2)
        with col1:
             display_story_segment("V12: Price Histogram", "Price Frequency with Filter.", "Market skewness and affordability 📉.")
             max_p = st.slider("Filter Max Price", 100000, 2000000, 1000000, step=50000, key="v12_slider")
             st.plotly_chart(vc.plot_price_histogram(df, max_price=max_p), use_container_width=True)
        with col2:
             display_story_segment("V13: Market Tier Donut", "Price Bands Proportion.", "Market accessibility 🥯.")
             st.plotly_chart(vc.plot_market_tier_donut(df), use_container_width=True)
        
        # Row 2: Variance
        st.markdown("#### Variability Analysis")
        display_story_segment("V14: County Variance", "Box Plots by County.", "Market heterogeneity 📊.")
        st.plotly_chart(vc.plot_county_variance_box(df), use_container_width=True)

        # Row 3: Comparisons
        col3, col4 = st.columns(2)
        with col3:
             display_story_segment("V15: New vs Second-Hand", "Violin Plot Comparison.", "New Build Premium 🏗️.")
             st.plotly_chart(vc.plot_new_vs_secondhand_violin(df), use_container_width=True)
        with col4:
             display_story_segment("V16: Temporal Ridgeline", "Density per Year.", "Bracket Creep 🐛 over time.")
             st.plotly_chart(vc.plot_temporal_ridgeline(df), use_container_width=True)

    # --- MODULE D: ATTRIBUTE CORRELATIONS ---
    elif selected_module == "Module D: Attribute Correlations":
        st.markdown("### 🏠 Property Attributes & Composition")

        # Row 1: Composition
        col1, col2 = st.columns(2)
        with col1:
            display_story_segment("V17: VAT Status", "New vs Existing Supply.", "New supply entering market 🏗️.")
            st.plotly_chart(vc.plot_vat_status_composition(df), use_container_width=True)
        with col2:
            display_story_segment("V20: Market Composition", "Sunburst Hierarchy.", "Deep drill-down 🎯.")
            st.plotly_chart(vc.plot_market_composition_sunburst(df), use_container_width=True)

        # Row 2: Size Analysis
        st.markdown("#### Size vs Value Analysis")
        col3, col4 = st.columns(2)
        with col3:
             display_story_segment("V18: Size Category Stacked", "Size Counts per Year.", "Changing morphology 🏠.")
             st.plotly_chart(vc.plot_size_category_stacked_bar(df), use_container_width=True)
        with col4:
             # V19 is Scatter Matrix, might be big
             display_story_segment("V19: Price vs Size", "Scatter Matrix.", "Correlation floor area 📏 vs value.")
             st.plotly_chart(vc.plot_price_vs_size_scatter_matrix(df), use_container_width=True)
        
        # Row 3: Multivariate
        st.markdown("#### Multivariate Flows")
        display_story_segment("V21: Parallel Coordinates", "Connecting variables.", "Common profiles and flows 🌊.")
        st.plotly_chart(vc.plot_parallel_coordinates(df), use_container_width=True)

    # --- MODULE E: PREDICTIVE MODELING ---
    elif selected_module == "Module E: Predictive Modeling":
        st.markdown("### 🔮 Future Market Forecast")
        st.markdown("""
        **Objective:** To predict future market trends and evaluate the importance of seasonality in real estate pricing.
        
        We compare two models:
        *   **ARIMA (AutoRegressive Integrated Moving Average):** A standard model that looks at past trends.
        *   **SARIMA (Seasonal ARIMA):** Adds a 'seasonality' component to account for recurring yearly patterns (e.g., the 'Spring Bloom').
        
        *Why this matters:* If SARIMA outperforms ARIMA, it confirms that market cycles are predictable, allowing for better timing of buying/selling decisions.
        """)
        st.info("Forecasting requires significant historical data. Best viewed with 'All' years selected.")
        
        display_story_segment("V22: SARIMA vs ARIMA", "Seasonal vs Standard ARIMA on hold-out.", "Model accuracy predictions 🔮.")

        if len(df) < 50:
             st.error("Insufficient data points for robust forecasting.")
        else:
             with st.spinner("Training models and generating forecast... This may take a moment ⏳"):
                 try:
                    train, test, sarima_yx, sarima_ci, arima_yx = run_forecasting_models(df)
                    st.plotly_chart(vc.plot_forecast(train, test, sarima_yx, sarima_ci, arima_yx), use_container_width=True)
                 except Exception as e:
                    st.error(f"Forecasting failed: {str(e)}")

    # --- MODULE F: CLUSTERING ANALYSIS ---
    elif selected_module == "Module F: Clustering Analysis":
        st.markdown("### 🧩 Advanced Market Segmentation")
        st.markdown("""
        **Objective:** To move beyond simple price bands and discover 'natural' groupings of properties based on their features (Price, Location, Type).
        """)
        
        with st.spinner("Running Clustering Algorithms..."):
             cluster_df, kmeans_model, fcm_model = run_clustering_models(df)

        if cluster_df.empty:
             st.error("Not enough data for clustering.")
        else:
            # Row 1: K-Means
            st.markdown("#### 1. K-Means Segmentation (Hard Clustering)")
            st.markdown("""
            *Intention:* To categorize the market into distinct, non-overlapping tiers (e.g., 'Economy', 'Mid-Market', 'Premium'). 
            This helps in understanding the primary market structure and identifying which segment a property definitely belongs to.
            """)
            col1, col2 = st.columns(2)
            with col1:
                display_story_segment("V23: K-Means Distribution", "Counts per Segment.", "Distinct tiers (Budget to Premium) 🏷️.")
                st.plotly_chart(vc.plot_cluster_distribution(cluster_df, 'Market_Segment', 'K-Means Market Segments'), use_container_width=True)
            with col2:
                display_story_segment("V24: Price vs Segment", "Box Plot by Segment.", "Validating segmentation 📊.")
                st.plotly_chart(vc.plot_price_vs_segment_box(cluster_df, 'Market_Segment', 'Price Distribution by Market Segment'), use_container_width=True)
                
            with st.expander("Show Segment Statistics"):
                st.dataframe(cluster_df.groupby('Market_Segment')['Price'].describe())

            # Row 2: Fuzzy C-Means
            st.markdown("#### 2. Fuzzy C-Means (Soft Clustering)")
            st.markdown("""
            *Intention:* Real estate isn't always black and white. A property might be 'mostly' Luxury but have some 'Mid-Market' characteristics.
            **Fuzzy Clustering** assigns probabilities (e.g., 70% Premium, 30% Mid-Market), allowing us to find 'bridge' properties or edge cases that K-Means might misclassify.
            """)
            col3, col4 = st.columns(2)
            with col3:
                 display_story_segment("V25: Fuzzy Distribution", "Counts per Fuzzy Segment.", "Soft membership assignments ☁️.")
                 st.plotly_chart(vc.plot_cluster_distribution(cluster_df, 'Fuzzy_Segment', 'Fuzzy C-Means Market Segments'), use_container_width=True)
            with col4:
                 display_story_segment("V26: Membership Strength", "Histogram of Probability.", "Ambiguous properties 🌫️.")
                 st.plotly_chart(vc.plot_fuzzy_membership_distribution(cluster_df), use_container_width=True)
