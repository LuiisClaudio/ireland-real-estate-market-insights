import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fcmeans import FCM
import dataframe_functions

def run_clustering_models(df):
    """
    Runs KMeans and Fuzzy C-Means clustering on the provided dataframe.
    
    Processing steps taken from Luis_Martins_CA1.ipynb:
    1. Filter for Sale_Year == 2025.
    2. Filter for Price < 5,000,000.
    3. Encode 'County'.
    4. Feature selection: ['Price', 'County_Encoded', 'Description_of_Property'].
    5. Scaling.
    6. KMeans (k=5).
    7. Fuzzy C-Means (c=5).
    
    Returns:
        df_clustering (pd.DataFrame): Dataframe with 'Market_Segment', 'Fuzzy_Segment' and 'Max_Membership'.
        kmeans_model: Trained KMeans model
        fuzzy_model: Trained FCM model
    """
    # Work on a copy
    data = df.copy()
    
    
    if data.empty:
        return data, None, None

    # Ensure clean data
    data = data.dropna(subset=['Price', 'County'])
    if 'Description_of_Property' not in data.columns:
         data['Description_of_Property'] = 0 # Default if missing

    
    if data.empty:
        return data, None, None

    # 3. Encode 'County'
    le = LabelEncoder()
    data['County_Encoded'] = le.fit_transform(data['County'])
    
    # 4. Feature Selection
    # Ensure Description_of_Property is numeric (0/1) as per clean_data
    # If clean_data was run, it should be numeric.
    features = ['Price', 'County_Encoded', 'Description_of_Property']
    
    # 5. Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    
    # 6. KMeans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    data['KMeans_Label'] = kmeans.fit_predict(data_scaled)
    
    # Assign Names to KMeans Clusters
    # Logic: Sort clusters by mean price to assign semantic labels
    # 0: Starter/Budget, 1: Affordable Family, 2: Mid-Range, 3: Upper-Mid, 4: Premium
    # Note: The cluster IDs (0-4) from KMeans are arbitrary, so we must sort them by price to map correctly.
    
    kmeans_means = data.groupby('KMeans_Label')['Price'].mean().sort_values()
    kmeans_mapping = {
        kmeans_means.index[0]: 'Starter/Budget',
        kmeans_means.index[1]: 'Affordable Family',
        kmeans_means.index[2]: 'Mid-Range',
        kmeans_means.index[3]: 'Upper-Mid',
        kmeans_means.index[4]: 'Premium'
    }
    data['Market_Segment'] = data['KMeans_Label'].map(kmeans_mapping)
    
    # 7. Fuzzy C-Means
    # fcmeans expects shape (n_samples, n_features)
    fcm = FCM(n_clusters=5, m=2, random_state=42, max_iter=150)
    fcm.fit(data_scaled)
    
    # Get hard labels and membership
    data['Fuzzy_Label'] = fcm.predict(data_scaled)
    data['Max_Membership'] = fcm.u.max(axis=1)
    
    # Assign Names to Fuzzy Clusters
    fuzzy_means = data.groupby('Fuzzy_Label')['Price'].mean().sort_values()
    fuzzy_mapping = {
        fuzzy_means.index[0]: 'Starter/Budget',
        fuzzy_means.index[1]: 'Affordable Family',
        fuzzy_means.index[2]: 'Mid-Range',
        fuzzy_means.index[3]: 'Upper-Mid',
        fuzzy_means.index[4]: 'Premium'
    }
    data['Fuzzy_Segment'] = data['Fuzzy_Label'].map(fuzzy_mapping)
    
    return data, kmeans, fcm


