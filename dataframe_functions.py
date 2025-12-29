import pandas as pd
import numpy as np


def load_data(file_path):
    # Define the file path
    file_path = 'PPR-ALL.csv'
    df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    return df

def clean_data(df):
    # Rename columns to be more Python-friendly (remove spaces, symbols)
    data = df.copy()
    data.columns = ['Date_of_Sale', 'Address', 'County', 'Eircode', 'Price', 
                    'Not_Full_Market_Price', 'VAT_Exclusive', 'Description_of_Property', 
                    'Property_Size_Description']

    # Clean the 'Price' column
    data['Price'] = data['Price'].astype(str).str.replace('€', '', regex=False).str.replace(',', '', regex=False).str.replace('', '', regex=False).str.strip()
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

    # Convert 'Date_of_Sale' to datetime objects
    data['Date_of_Sale'] = pd.to_datetime(data['Date_of_Sale'], format='%d/%m/%Y')

    # Extract 'Year' and 'Month' as new features
    data['Sale_Year'] = data['Date_of_Sale'].dt.year
    data['Sale_Month'] = data['Date_of_Sale'].dt.month

    # Clean and impute 'Description_of_Property'
    # Replace empty/whitespace strings with 'Unknown'
    data['Description_of_Property'] = data['Description_of_Property'].replace(['""', '', ' '], 'Unknown')
    # Fill standard NaNs with 'Unknown' as well
    data['Description_of_Property'] = data['Description_of_Property'].fillna('Unknown')
    # There are two variations of 'New Dwelling...' in the data, let's standardize them.
    data['Description_of_Property'] = data['Description_of_Property'].replace({
        'New Dwelling house /Apartment': 'New Dwelling',
        'Second-Hand Dwelling house /Apartment': 'Second-Hand Dwelling',
        'Teach/Árasán Cónaithe Nua': 'New Dwelling',
        'Teach/?ras?n C?naithe Nua': 'New Dwelling',
        'Teach/Árasán Cónaithe Atháimhe': 'Second-Hand Dwelling'
    })

        
    # Clean and impute 'County'
    data['County'] = data['County'].fillna('Unknown')

    # Map binary columns to 1/0
    data['Not_Full_Market_Price'] = data['Not_Full_Market_Price'].map({'Yes': 1, 'No': 0})
    data['VAT_Exclusive'] = data['VAT_Exclusive'].map({'Yes': 1, 'No': 0})
    data['Description_of_Property'] = data['Description_of_Property'].map({'New Dwelling': 1, 'Second-Hand Dwelling': 0})
    
    return data

def csv_dump(df, file_name):
    df = df.sample(20000, random_state=42).copy()
    df.to_csv(file_name, index=False)

# Do a main
if __name__ == "__main__":
    df = load_data('PPR-ALL.csv')
    cleaned_df = clean_data(df)
    csv_dump(cleaned_df, 'cleaned_real_estate_data.csv')
    print(cleaned_df.head())
