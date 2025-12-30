import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st

# --- DESIGN SYSTEM ---
# Premium Teal/Slate Color Palette
PREMIUM_COLOR_SEQUENCE = [
    '#264653', # Charcoal/Deep Teal (Primary)
    '#2a9d8f', # Persian Green (Secondary)
    '#e9c46a', # Saffron (Highlight)
    '#f4a261', # Sandy Brown
    '#e76f51', # Burnt Sienna (Alert/Important)
    '#1A2C42', # Deep Indices
    '#8AB17D', # Sage
    '#6d597a'  # Dusty Purple
]

# Continuous Scale for Heatmaps/Choropleths
PREMIUM_SCALE = px.colors.sequential.Tealgrn # Elegant Green/Teal gradient

# Standard Layout Template
def update_layout_premium(fig, title):
    fig.update_layout(
        title=title,
        template='plotly_white',
        colorway=PREMIUM_COLOR_SEQUENCE,
        xaxis_title=None,
        yaxis_title=None,
        font=dict(family="Inter, sans-serif"),
        hovermode="x unified"
    )
    return fig

# Helper to map counties to provinces
def get_province(county):
    leinster = ['Carlow', 'Dublin', 'Kildare', 'Kilkenny', 'Laois', 'Longford', 'Louth', 'Meath', 'Offaly', 'Westmeath', 'Wexford', 'Wicklow']
    munster = ['Clare', 'Cork', 'Kerry', 'Limerick', 'Tipperary', 'Waterford']
    connacht = ['Galway', 'Leitrim', 'Mayo', 'Roscommon', 'Sligo']
    ulster = ['Cavan', 'Donegal', 'Monaghan']
    
    if county in leinster: return 'Leinster'
    if county in munster: return 'Munster'
    if county in connacht: return 'Connacht'
    if county in ulster: return 'Ulster'
    return 'Unknown'

# Module A: Temporal Dynamics

def plot_market_velocity_kpi(df):
    """V1: Market Velocity KPI Banner"""
    total_volume = df['Price'].sum()
    median_price = df['Price'].median()
    total_units = len(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transaction Volume", f"€{total_volume:,.0f}")
    col2.metric("Median Transaction Price", f"€{median_price:,.0f}")
    col3.metric("Total Units Sold", f"{total_units:,}")

def plot_median_price_trend(df):
    """V2: The Inflation Tracker - Median Price Line Chart"""
    # Group by Sale_Year and Sale_Month to create a date for plotting
    df_grouped = df.groupby(['Sale_Year', 'Sale_Month'])['Price'].median().reset_index()
    df_grouped['Date'] = df_grouped.apply(lambda row: f"{int(row['Sale_Year'])}-{int(row['Sale_Month']):02d}-01", axis=1)
    
    fig = px.line(df_grouped, x='Date', y='Price', title='Median Price Trend Over Time', 
                  color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    
    # Fill area under line for "Premium" feel
    fig.update_traces(fill='tozeroy', line=dict(width=3))
    return update_layout_premium(fig, 'Median Price Trend Over Time')

def plot_regional_divergence(df):
    """V3: Regional Divergence Multi-Line Chart"""
    top_counties = df['County'].value_counts().head(5).index.tolist()
    df_filtered = df[df['County'].isin(top_counties)]
    df_grouped = df_filtered.groupby(['Sale_Year', 'County'])['Price'].median().reset_index()
    
    fig = px.line(df_grouped, x='Sale_Year', y='Price', color='County', 
                  title='Median Price Trends by Top Counties',
                  color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, 'Median Price Trends by Top Counties')

def plot_seasonality(df):
    """V4: Seasonality Bar Chart"""
    df_grouped = df.groupby('Sale_Month').size().reset_index(name='Transactions')
    fig = px.bar(df_grouped, x='Sale_Month', y='Transactions', title='Seasonality of Transactions',
                 color='Transactions', color_continuous_scale=PREMIUM_SCALE)
    return update_layout_premium(fig, 'Seasonality of Transactions')

def plot_market_heatmap(df):
    """V5: Market Heatmap"""
    df_grouped = df.groupby(['Sale_Year', 'Sale_Month']).size().reset_index(name='Volume')
    fig = px.density_heatmap(df_grouped, x='Sale_Month', y='Sale_Year', z='Volume', 
                             title='Market Activity Heatmap',
                             nbinsx=12, nbinsy=len(df['Sale_Year'].unique()),
                             color_continuous_scale=PREMIUM_SCALE)
    return update_layout_premium(fig, 'Market Activity Heatmap')

def plot_volume_price_correlation(df):
    """V6: Volume-Price Correlation Dual-Axis Chart"""
    df_grouped = df.groupby('Sale_Year').agg({'Price': 'median', 'Date_of_Sale': 'count'}).rename(columns={'Date_of_Sale': 'Volume'}).reset_index()
    
    fig = go.Figure()
    # Use Primary color for Price (Line) and Secondary/Muted for Volume (Bar)
    fig.add_trace(go.Bar(x=df_grouped['Sale_Year'], y=df_grouped['Volume'], name='Volume', yaxis='y2', 
                         opacity=0.3, marker_color=PREMIUM_COLOR_SEQUENCE[1]))
    fig.add_trace(go.Scatter(x=df_grouped['Sale_Year'], y=df_grouped['Price'], name='Median Price', yaxis='y1',
                             line=dict(color=PREMIUM_COLOR_SEQUENCE[0], width=3)))
    
    fig.update_layout(
        title='Volume vs Price Correlation',
        template='plotly_white',
        yaxis=dict(title='Median Price'),
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        colorway=PREMIUM_COLOR_SEQUENCE,
        font=dict(family="Inter, sans-serif")
    )
    return fig

# Module B: Geospatial Intelligence

def plot_premium_postcode_ranking(df):
    """V7: The Premium Postcode Ranking"""
    
    df['RoutingKey'] = df['Eircode'].astype(str).str[:3]
    valid_keys = df[df['RoutingKey'].str.len() == 3]
    
    if valid_keys.empty:
         df_grouped = df.groupby('County')['Price'].median().sort_values(ascending=True).tail(20)
         title = 'Top Areas by Median Price (County Fallback)'
         y_col = df_grouped.index
    else:
        df_grouped = valid_keys.groupby('RoutingKey')['Price'].median().sort_values(ascending=True).tail(20)
        title = 'Top Routing Keys by Median Price'
        y_col = df_grouped.index
    
    fig = px.bar(x=df_grouped.values, y=y_col, orientation='h', title=title,
                 color=df_grouped.values, color_continuous_scale=PREMIUM_SCALE)
    fig.update_layout(xaxis_title="Median Price (€)", yaxis_title="Area")
    return update_layout_premium(fig, title)

COUNTY_CENTROIDS = {
    'Carlow': (52.70, -6.80), 'Cavan': (53.99, -7.36), 'Clare': (52.90, -9.00),
    'Cork': (51.90, -8.47), 'Donegal': (54.91, -7.75), 'Dublin': (53.35, -6.26),
    'Galway': (53.27, -9.05), 'Kerry': (52.15, -9.56), 'Kildare': (53.16, -6.82),
    'Kilkenny': (52.65, -7.24), 'Laois': (53.03, -7.30), 'Leitrim': (54.12, -8.00),
    'Limerick': (52.66, -8.62), 'Longford': (53.72, -7.80), 'Louth': (53.89, -6.49),
    'Mayo': (53.90, -9.26), 'Meath': (53.60, -6.65), 'Monaghan': (54.24, -6.97),
    'Offaly': (53.23, -7.71), 'Roscommon': (53.76, -8.24), 'Sligo': (54.27, -8.47),
    'Tipperary': (52.47, -7.90), 'Waterford': (52.25, -7.11), 'Westmeath': (53.53, -7.46),
    'Wexford': (52.33, -6.46), 'Wicklow': (53.00, -6.30)
}

def plot_national_price_choropleth(df):
    """V8: National Price Scatter Map (Replaced Choropleth)"""
    df_grouped = df.groupby('County')['Price'].mean().reset_index()
    df_grouped['Coords'] = df_grouped['County'].map(COUNTY_CENTROIDS)
    df_grouped = df_grouped.dropna(subset=['Coords'])
    
    if df_grouped.empty: return None
        
    df_grouped['Latitude'] = df_grouped['Coords'].apply(lambda x: x[0])
    df_grouped['Longitude'] = df_grouped['Coords'].apply(lambda x: x[1])
    
    fig = px.scatter_mapbox(
        df_grouped,
        lat="Latitude", lon="Longitude", size="Price", color="Price",
        color_continuous_scale=PREMIUM_SCALE,
        size_max=30, zoom=5.5, center={"lat": 53.4, "lon": -7.9},
        mapbox_style="carto-positron",
        hover_name='County',
        title="National Mean Price by County"
    )
    return update_layout_premium(fig, "National Mean Price by County")

def plot_hyper_local_scatter(df):
    """V9: Hyper-Local Scatter Mapbox"""
    plot_df = df.copy()
    
    if 'Latitude' not in plot_df.columns or 'Longitude' not in plot_df.columns:
        plot_df['Coords'] = plot_df['County'].map(COUNTY_CENTROIDS)
        plot_df = plot_df.dropna(subset=['Coords'])
        if plot_df.empty: return None
        plot_df['Latitude'] = plot_df['Coords'].apply(lambda x: x[0])
        plot_df['Longitude'] = plot_df['Coords'].apply(lambda x: x[1])
        np.random.seed(42)
        plot_df['Latitude'] += np.random.normal(0, 0.04, size=len(plot_df))
        plot_df['Longitude'] += np.random.normal(0, 0.04, size=len(plot_df))
    
    fig = px.scatter_mapbox(plot_df, lat="Latitude", lon="Longitude", color="Price", size="Price",
                      color_continuous_scale=PREMIUM_SCALE, size_max=15, zoom=6,
                      mapbox_style="carto-positron",
                      title="Hyper-Local Sales Scatter")
    return update_layout_premium(fig, "Hyper-Local Sales Scatter")

def plot_urban_density_hexagon(df):
    """V10: Urban Density Hexagon Layer (PyDeck - No Theme Update needed but ensuring colors align)"""
    plot_df = df.copy()

    if 'Latitude' not in plot_df.columns or 'Longitude' not in plot_df.columns:
         plot_df['Coords'] = plot_df['County'].map(COUNTY_CENTROIDS)
         plot_df = plot_df.dropna(subset=['Coords'])
         if plot_df.empty: return None
         plot_df['Latitude'] = plot_df['Coords'].apply(lambda x: x[0])
         plot_df['Longitude'] = plot_df['Coords'].apply(lambda x: x[1])
         np.random.seed(42)
         plot_df['Latitude'] += np.random.normal(0, 0.04, size=len(plot_df))
         plot_df['Longitude'] += np.random.normal(0, 0.04, size=len(plot_df))

    # Using a teal color range for consistency
    layer = pdk.Layer(
        'HexagonLayer',
        plot_df,
        get_position='[Longitude, Latitude]',
        auto_highlight=True,
        elevation_scale=50,
        pickable=True,
        elevation_range=[0, 3000],
        extruded=True,
        coverage=1,
        color_range=[
            [237, 248, 251],
            [178, 226, 226],
            [102, 194, 164],
            [44, 162, 95],
            [0, 109, 44]
        ]
    )
    view_state = pdk.ViewState(
        longitude=plot_df['Longitude'].mean(),
        latitude=plot_df['Latitude'].mean(),
        zoom=6,
        min_zoom=5,
        max_zoom=15,
        pitch=40.5,
        bearing=-27.36
    )
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Count: {elevationValue}"})
    return deck

def plot_provincial_treemap(df):
    """V11: Provincial Treemap"""
    df['Province'] = df['County'].apply(get_province)
    df_grouped = df.groupby(['Province', 'County'])['Price'].agg(['count', 'median']).reset_index()
    df_grouped.columns = ['Province', 'County', 'Volume', 'Median_Price']
    
    fig = px.treemap(df_grouped, path=['Province', 'County'], values='Volume', color='Median_Price',
                     title='Provincial Market Treemap',
                     color_continuous_scale=PREMIUM_SCALE)
    return update_layout_premium(fig, 'Provincial Market Treemap')

# Module C: Distribution and Affordability

def plot_price_histogram(df, max_price=None):
    """V12: Price Histogram with Outlier Clipper"""
    if max_price:
        df = df[df['Price'] <= max_price]
    fig = px.histogram(df, x='Price', nbins=50, title='Price Distribution',
                       color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, 'Price Distribution')

def plot_market_tier_donut(df):
    """V13: Market Tier Donut Chart"""
    conditions = [
        (df['Price'] < 320000),
        (df['Price'] >= 320000) & (df['Price'] <= 400000),
        (df['Price'] > 400000)
    ]
    choices = ['Starter (<€320k)', 'Middle Market (€320k-€400k)', 'Premium (>€400k)']
    df['Tier'] = np.select(conditions, choices, default='Unknown')
    
    df_grouped = df['Tier'].value_counts().reset_index()
    df_grouped.columns = ['Tier', 'Count']
    
    fig = px.pie(df_grouped, values='Count', names='Tier', hole=0.4, title='Market Segments',
                 color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, 'Market Segments')

def plot_county_variance_box(df):
    """V14: County Variance Box Plots"""
    fig = px.box(df, x='County', y='Price', title='Price Variance by County',
                 color='County', color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    fig.update_layout(showlegend=False)
    return update_layout_premium(fig, 'Price Variance by County')

def plot_new_vs_secondhand_violin(df):
    """V15: New vs. Second-Hand Violin Plot"""
    df['Property Type'] = df['Description_of_Property'].map({1: 'New', 0: 'Second Hand', 'Unknown': 'Unknown'})
    df_filtered = df[df['Property Type'] != 'Unknown']
    
    fig = px.violin(df_filtered, x='Property Type', y='Price', box=True, points="all", 
                    title='New vs Second-Hand Price Distribution',
                    color='Property Type', color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, 'New vs Second-Hand Price Distribution')

def plot_temporal_ridgeline(df):
    """V16: Temporal Ridgeline Plot"""
    fig = px.violin(df, x='Price', y='Sale_Year', orientation='h', 
                    title='Price Distribution Over Time',
                    color='Sale_Year', color_discrete_sequence=PREMIUM_SCALE) # Discrete scale from sequential
    return update_layout_premium(fig, 'Price Distribution Over Time')

# Module D: Attribute Correlations

def plot_vat_status_composition(df):
    """V17: VAT Status Composition"""
    df_grouped = df['VAT_Exclusive'].value_counts().reset_index()
    df_grouped.columns = ['VAT Exclusive', 'Count']
    df_grouped['VAT Exclusive'] = df_grouped['VAT Exclusive'].map({1: 'Yes', 0: 'No'})
    
    fig = px.pie(df_grouped, values='Count', names='VAT Exclusive', title='VAT Status Composition',
                 color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, 'VAT Status Composition')

def plot_size_category_stacked_bar(df):
    """V18: Size Category Stacked Bar"""
    df_grouped = df.groupby(['Sale_Year', 'Property_Size_Description']).size().reset_index(name='Count')
    fig = px.bar(df_grouped, x='Sale_Year', y='Count', color='Property_Size_Description', 
                 title='Property Size Distribution by Year',
                 color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, 'Property Size Distribution by Year')

def plot_price_vs_size_scatter_matrix(df):
    """V19: Price vs. Size Scatter Matrix"""
    fig = px.strip(df, x='Property_Size_Description', y='Price', title='Price vs Property Size Category',
                   color='Property_Size_Description', color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, 'Price vs Property Size Category')

def plot_market_composition_sunburst(df):
    """V20: Market Composition Sunburst"""
    df['Province'] = df['County'].apply(get_province)
    df['Property Type'] = df['Description_of_Property'].map({1: 'New', 0: 'Second Hand'})
    df = df.fillna('Unknown')
    
    fig = px.sunburst(df, path=['Province', 'County', 'Property Type'], values='Price', 
                      title='Market Composition',
                      color='Price', color_continuous_scale=PREMIUM_SCALE)
    return update_layout_premium(fig, 'Market Composition')

def plot_parallel_coordinates(df):
    """V21: Multivariate Parallel Coordinates"""
    df_sample = df
    df_sample['County_Code'] = df_sample['County'].astype('category').cat.codes
    
    fig = px.parallel_coordinates(df_sample, 
                                  dimensions=['County_Code', 'Sale_Year', 'Sale_Month', 'Price'],
                                  color="Price", 
                                  title='Multivariate Parallel Coordinates',
                                  color_continuous_scale=PREMIUM_SCALE)
    return update_layout_premium(fig, 'Multivariate Parallel Coordinates')

# Module E: Predictive Modeling

def plot_forecast(train_data, test_data, sarima_mean, sarima_conf_int, arima_mean=None):
    """V22: SARIMA vs ARIMA Forecast Plot"""
    fig = go.Figure()

    # Themes: Charcoal, Teal, Saffron
    # Training Data - Slate
    fig.add_trace(go.Scatter(
        x=train_data.index, y=train_data, mode='lines+markers', name='Training Data',
        line=dict(color=PREMIUM_COLOR_SEQUENCE[5], width=2) # Dusty Purple
    ))

    # Actual Test Data - Sage Green
    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data, mode='lines+markers', name='Actual Test Data',
        line=dict(color=PREMIUM_COLOR_SEQUENCE[6], width=2)
    ))

    # SARIMA Forecast - Primary Teal
    fig.add_trace(go.Scatter(
        x=test_data.index, y=sarima_mean, mode='lines+markers', name='SARIMA Forecast',
        line=dict(color=PREMIUM_COLOR_SEQUENCE[0], dash='dash', width=2)
    ))

    # SARIMA Confidence Interval
    fig.add_trace(go.Scatter(
        x=test_data.index, y=sarima_conf_int.iloc[:, 1], # Upper
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=test_data.index, y=sarima_conf_int.iloc[:, 0], # Lower
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(38, 70, 83, 0.2)', # Transparent Primary Teal
        name='95% Confidence Interval (SARIMA)', hoverinfo='skip'
    ))

    # ARIMA Forecast (Optional) - Sandy Brown
    if arima_mean is not None:
        fig.add_trace(go.Scatter(
            x=test_data.index, y=arima_mean, mode='lines', name='ARIMA Forecast',
            line=dict(color=PREMIUM_COLOR_SEQUENCE[3], dash='dot', width=2)
        ))

    return update_layout_premium(fig, 'SARIMA vs ARIMA Model Forecast')

# Module F: Clustering Analysis

def plot_cluster_distribution(df, segment_col, title):
    """V23/V25: Cluster Distribution Bar Chart"""
    if segment_col not in df.columns: return None
    df_grouped = df[segment_col].value_counts().reset_index()
    df_grouped.columns = ['Market Segment', 'Count']
    fig = px.bar(df_grouped, x='Market Segment', y='Count', title=title, 
                 color='Market Segment', color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, title)

def plot_price_vs_segment_box(df, segment_col, title):
    """V24: Price vs Market Segment Box Plot"""
    if segment_col not in df.columns: return None
    fig = px.box(df, x=segment_col, y='Price', title=title, 
                 color=segment_col, color_discrete_sequence=PREMIUM_COLOR_SEQUENCE)
    return update_layout_premium(fig, title)

def plot_fuzzy_membership_distribution(df):
    """V26: Fuzzy Membership Distribution"""
    if 'Max_Membership' not in df.columns: return None
    fig = px.histogram(df, x='Max_Membership', nbins=50, 
                       title='Fuzzy C-Means Membership Strength Distribution',
                       labels={'Max_Membership': 'Membership Probability'},
                       color_discrete_sequence=[PREMIUM_COLOR_SEQUENCE[1]]) # Secondary Theme Color
    return update_layout_premium(fig, 'Fuzzy C-Means Membership Strength Distribution')
