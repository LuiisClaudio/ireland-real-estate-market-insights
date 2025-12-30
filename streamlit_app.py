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

    lower_bound = int(Q_inf - 1.5 * IQR)
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

# Navigation Structure
NAV_STRUCT = {
    "Module A: Temporal Dynamics": {
        "V1: Market Velocity KPI": vc.plot_market_velocity_kpi,
        "V2: Median Price Trend": vc.plot_median_price_trend,
        "V3: Regional Divergence": vc.plot_regional_divergence,
        "V4: Seasonality": vc.plot_seasonality,
        "V5: Market Heatmap": vc.plot_market_heatmap,
        "V6: Volume-Price Correlation": vc.plot_volume_price_correlation
    },
    "Module B: Geospatial Intelligence": {
        "V7: Premium Postcode Ranking": vc.plot_premium_postcode_ranking,
        "V8: National Price Choropleth": vc.plot_national_price_choropleth, # Placeholder
        "V9: Hyper-Local Scatter Mapbox": vc.plot_hyper_local_scatter,
        "V10: Urban Density Hexagon": vc.plot_urban_density_hexagon,
        "V11: Provincial Treemap": vc.plot_provincial_treemap
    },
    "Module C: Distribution & Affordability": {
        "V12: Price Histogram": vc.plot_price_histogram,
        "V13: Market Tier Donut": vc.plot_market_tier_donut,
        "V14: County Variance Box Plots": vc.plot_county_variance_box,
        "V15: New vs Second-Hand Violin": vc.plot_new_vs_secondhand_violin,
        "V16: Temporal Ridgeline": vc.plot_temporal_ridgeline
    },
    "Module D: Attribute Correlations": {
        "V17: VAT Status Composition": vc.plot_vat_status_composition,
        "V18: Size Category Stacked Bar": vc.plot_size_category_stacked_bar,
        "V19: Price vs Size Scatter Matrix": vc.plot_price_vs_size_scatter_matrix,
        "V20: Market Composition Sunburst": vc.plot_market_composition_sunburst,
        "V21: Multivariate Parallel Coordinates": vc.plot_parallel_coordinates
    },
    "Module E: Predictive Modeling": {
        "V22: SARIMA vs ARIMA Forecast": vc.plot_forecast
    },
    "Module F: Clustering Analysis": {
        "V23: K-Means Cluster Distribution": vc.plot_cluster_distribution,
        "V24: Price vs Market Segment (K-Means)": vc.plot_price_vs_segment_box,
        "V25: Fuzzy Cluster Distribution": vc.plot_cluster_distribution,
        "V26: Fuzzy Membership Strength": vc.plot_fuzzy_membership_distribution
    }
}

# Visualization Information
VIZ_INFO = {
    "V1: Market Velocity KPI": {
        "Logic": "Calculates dynamic aggregates based on the user's filtered selection.",
        "Insight": "Provides an immediate 'pulse' 💓 of the selected market segment."
    },
    "V2: Median Price Trend": {
        "Logic": "Plots the median Price filtered by Sale_Month and Sale_Year.",
        "Insight": "Visualizes the macro-trend 📈, highlighting the recovery trajectory and inflation peaks."
    },
    "V3: Regional Divergence": {
        "Logic": "Comparison of the top 5 counties by volume.",
        "Insight": "Demonstrates the decoupling of the Dublin market 🏙️ from the rest of the country."
    },
    "V4: Seasonality": {
        "Logic": "Aggregates total transaction counts by Sale_Month (1-12).",
        "Insight": "Reveals the 'Spring Bloom' 🌸 (Q2 spike) and 'Winter Lull' ❄️."
    },
    "V5: Market Heatmap": {
        "Logic": "2D matrix of Month vs Year with color intensity as Volume.",
        "Insight": "Reveals structural breaks 🧱, like the impact of COVID-19 lockdowns."
    },
    "V6: Volume-Price Correlation": {
        "Logic": "Overlays Median Price line and Volume bar chart.",
        "Insight": "Analyzes the relationship between supply 📦 and price 💰."
    },
    "V7: Premium Postcode Ranking": {
        "Logic": "Ranking top 20 Areas by median price.",
        "Insight": "Identifies the market's 'Premium' tiers 💎 (e.g., D4, Greystones)."
    },
    "V8: National Price Choropleth": {
        "Logic": "Choropleth map coloring counties by Median Price.",
        "Insight": "Provides a macro-spatial view of the 'East-West Divide' 🗺️."
    },
    "V9: Hyper-Local Scatter Mapbox": {
        "Logic": "Scatter plot on Mapbox using synthesized coordinates.",
        "Insight": "Visualizes local density 📍 and price distribution.",
    },
    "V10: Urban Density Hexagon": {
        "Logic": "3D Hexagon layer representing transaction count.",
        "Insight": "Identifies 'hotspots' 🔥 of activity in urban centers."
    },
    "V11: Provincial Treemap": {
        "Logic": "Hierarchical view: Province -> County.",
        "Insight": "Shows the relative weight ⚖️ of markets (e.g., Dublin dominance)."
    },
    "V12: Price Histogram": {
        "Logic": "Frequency distribution of Price with outlier filter.",
        "Insight": "Reveals the skewness 📉 of the market and mass-market affordability."
    },
    "V13: Market Tier Donut": {
        "Logic": "Proportion of stock in price bands (<€320k, etc).",
        "Insight": "Summarizes market accessibility and affordability 🥯."
    },
    "V14: County Variance Box Plots": {
        "Logic": "Box-and-whisker diagrams of prices by county.",
        "Insight": "Highlights market heterogeneity and variance 📊 within counties."
    },
    "V15: New vs Second-Hand Violin": {
        "Logic": "Violin plot comparing New vs Second-Hand prices.",
        "Insight": "Demonstrates the 'New Build Premium' 🏗️."
    },
    "V16: Temporal Ridgeline": {
        "Logic": "Stacked density plots per year.",
        "Insight": "Visualizes 'Bracket Creep' 🐛 and distribution shifts over time."
    },
    "V17: VAT Status Composition": {
        "Logic": "Breakdown of VAT Exclusive transactions.",
        "Insight": "Monitors new supply entering the market 🏗️."
    },
    "V18: Size Category Stacked Bar": {
        "Logic": "Counts of Property Size Description per year.",
        "Insight": "Tracks the changing morphology 🏠 of Irish housing."
    },
    "V19: Price vs Size Scatter Matrix": {
        "Logic": "Price against Size Category.",
        "Insight": "Validates correlation between floor area 📏 and value."
    },
    "V20: Market Composition Sunburst": {
        "Logic": "Radial hierarchy: Province -> County -> Type.",
        "Insight": "Allows deep drill-down 🎯 into market composition."
    },
    "V21: Multivariate Parallel Coordinates": {
        "Logic": "Connects variables (County, Size, VAT, Price) with lines.",
        "Insight": "Reveals common profiles and flows 🌊 across attributes."
    },
    "V22: SARIMA vs ARIMA Forecast": {
        "Logic": "Comparison of Seasonal ARIMA vs Standard ARIMA forecasts on a hold-out test set.",
        "Insight": "Evaluates model accuracy and the impact of seasonality on market predictions 🔮."
    },
    "V23: K-Means Cluster Distribution": {
        "Logic": "K-Means clustering (k=5) on Price, Location, and Type.",
        "Insight": "Segments the market into distinct tiers (Budget to Premium) 🏷️."
    },
    "V24: Price vs Market Segment (K-Means)": {
        "Logic": "Box plot of Price distribution across identified K-Means clusters.",
        "Insight": "Validates the segmentation by showing distinct price bands 📊."
    },
    "V25: Fuzzy Cluster Distribution": {
        "Logic": "Fuzzy C-Means clustering allowing soft membership.",
        "Insight": "Compares hard vs soft clustering assignments ☁️."
    },
    "V26: Fuzzy Membership Strength": {
        "Logic": "Histogram of maximum membership probabilities.",
        "Insight": "Reveals properties that are 'ambiguous' or between segments 🌫️."
    }
}

# Sidebar Navigation
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
selected_module = st.sidebar.selectbox("Select Module", list(NAV_STRUCT.keys()))
selected_viz_name = st.sidebar.radio("Select Visualization", list(NAV_STRUCT[selected_module].keys()))

# Main Dashboard
st.title("🇮🇪 Irish Real Estate Market Insights")
st.markdown(f"## **{selected_module}**") 
st.markdown(f"### *{selected_viz_name}*")

# Display Info
if selected_viz_name in VIZ_INFO:
    info = VIZ_INFO[selected_viz_name]
    with st.expander("ℹ️ visualization Logic & Insight", expanded=True):
        st.markdown(f"**Logic:** {info['Logic']}")
        st.markdown(f"**Insight:** {info['Insight']}")

if not df.empty:
    viz_func = NAV_STRUCT[selected_module][selected_viz_name]

    # Special handling for V1 (KPI) which returns nothing but renders inside function
    if selected_viz_name == "V1: Market Velocity KPI":
        viz_func(df)
        
    # Special handling for V12 (needs extra widget)
    elif selected_viz_name == "V12: Price Histogram":
        max_price_filter = st.slider("Filter Max Price for Histogram", 100000, 2000000, 1000000, step=50000)
        fig = viz_func(df, max_price=max_price_filter)
        st.plotly_chart(fig, use_container_width=True)

    # Special handling for Maps (V9, V10)
    elif selected_viz_name in ["V9: Hyper-Local Scatter Mapbox", "V10: Urban Density Hexagon"]:
        st.info("Note: Latitude/Longitude data is required for this visualization.")
        chart = viz_func(df)
        if chart:
            if selected_viz_name == "V9: Hyper-Local Scatter Mapbox":
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.pydeck_chart(chart)
        else:
            st.warning(f"Geolocation data not available for {selected_viz_name}.")

    # Special handling for V8 (Placeholder)
    elif selected_viz_name == "V8: National Price Choropleth":
        st.warning("GeoJSON data required for Choropleth map.")
        chart = viz_func(df)
        if chart: st.plotly_chart(chart, use_container_width=True)

    # Special handling for V22 Forecast
    elif selected_viz_name == "V22: SARIMA vs ARIMA Forecast":
        if df.empty:
            st.warning("Not enough data for forecasting.")
        else:
            with st.spinner("Training models and generating forecast... This may take a moment ⏳"):
                 # Check data sufficiency
                 if len(df) < 50: # Arbitrary small number check
                     st.error("Insufficient data points for robust forecasting.")
                 elif (df['Date'].max() - df['Date'].min()).days < 730:
                     st.error("Insufficient data duration. Please select a range of at least 2 years.")
                 else:
                     try:
                        train, test, sarima_yx, sarima_ci, arima_yx = run_forecasting_models(df)
                        fig = viz_func(train, test, sarima_yx, sarima_ci, arima_yx)
                        st.plotly_chart(fig, use_container_width=True)
                     except Exception as e:
                        st.error(f"Forecasting failed: {str(e)}")

    # Special handling for Module F (Clustering)
    elif selected_module == "Module F: Clustering Analysis":
        with st.spinner("Running Clustering Algorithms on Selected Data..."):
             # Use the filtered dataframe 'df' instead of 'df_raw'
             cluster_df, kmeans_model, fcm_model = run_clustering_models(df)
             
             if cluster_df.empty:
                 st.error("Not enough data available for clustering analysis with current filters.")
             else:
                 if selected_viz_name == "V23: K-Means Cluster Distribution":
                     fig = viz_func(cluster_df, 'Market_Segment', 'K-Means Market Segments')
                 elif selected_viz_name == "V24: Price vs Market Segment (K-Means)":
                     fig = viz_func(cluster_df, 'Market_Segment', 'Price Distribution by Market Segment (K-Means)')
                 elif selected_viz_name == "V25: Fuzzy Cluster Distribution":
                     fig = viz_func(cluster_df, 'Fuzzy_Segment', 'Fuzzy C-Means Market Segments')
                 elif selected_viz_name == "V26: Fuzzy Membership Strength":
                     fig = viz_func(cluster_df)
                 
                 if fig: st.plotly_chart(fig, use_container_width=True)
                 
                 # Display stats table
                 if selected_viz_name == "V24: Price vs Market Segment (K-Means)":
                     st.subheader("Segment Statistics")
                     stats = cluster_df.groupby('Market_Segment')['Price'].describe()
                     st.dataframe(stats)

    # Standard Plotly Charts
    else:
        fig = viz_func(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please adjust filters to view data.")
