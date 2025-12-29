import streamlit as st
import pandas as pd
import plotly.express as px
import visualization_code as vc

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

    # Apply Filters
    df = df_raw.copy()
    df = df[(df['Sale_Year'] >= selected_years[0]) & (df['Sale_Year'] <= selected_years[1])]
    
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
    }
}

# Sidebar Navigation
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
selected_module = st.sidebar.selectbox("Select Module", list(NAV_STRUCT.keys()))
selected_viz_name = st.sidebar.radio("Select Visualization", list(NAV_STRUCT[selected_module].keys()))

# Main Dashboard
st.title("🇮🇪 Irish Real Estate Market Insights")
st.markdown(f"**{selected_module}** > *{selected_viz_name}*")

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

    # Standard Plotly Charts
    else:
        fig = viz_func(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please adjust filters to view data.")
