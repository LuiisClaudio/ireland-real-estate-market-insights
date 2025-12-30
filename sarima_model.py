import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import dataframe_functions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from dataframe_functions import load_data, clean_data
from statsmodels.tsa.arima.model import ARIMA

def run_forecasting_models(df):
    """
    Fits SARIMA and ARIMA models and returns the forecasts.
    Fits models on the provided dataframe.
    """
    # Prepare Time Series (Monthly Average)
    # Check if 'Date' index exists or set it
    if 'Date' in df.columns:
        ts = df.set_index('Date')['Price'].resample('MS').mean().interpolate(method='linear')
    else:
         # Fallback if Date missing (shouldn't happen given load_data)
        ts = df.set_index('Date_of_Sale')['Price'].resample('MS').mean().interpolate(method='linear')
    # Split Train/Test (95/5)
    train_size = int(len(ts) * 0.95)
    train_data = ts[:train_size]
    test_data = ts[train_size:]
    
    # SARIMA (4, 1, 1) x (1, 1, 1, 12)
    sarima_model = SARIMAX(train_data, 
                           order=(4, 1, 1), 
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_fitted = sarima_model.fit(disp=False)
    
    sarima_forecast = sarima_fitted.get_forecast(steps=len(test_data))
    sarima_mean = sarima_forecast.predicted_mean
    sarima_conf_int = sarima_forecast.conf_int()
    
    # ARIMA (4, 1, 1)
    arima_model = ARIMA(train_data, order=(4, 1, 1))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.get_forecast(steps=len(test_data))
    arima_mean = arima_forecast.predicted_mean
    
    return train_data, test_data, sarima_mean, sarima_conf_int, arima_mean

def analyze_sarima():
    print("Loading and cleaning data using dataframe_functions")
    try:
        # Load and clean the data using functions from dataframe_functions
        raw_df = dataframe_functions.load_data('PPR-ALL.csv')
        df = dataframe_functions.clean_data(raw_df)
    except Exception as e:
        print(f"Error loading/cleaning data: {e}")
        return
    # Create Time Series: Monthly Average Price
    print("Preparing time series")
    # Using 'MS' is generally safer for plotting alignment
    ts = df.set_index('Date_of_Sale')['Price'].resample('MS').mean()
    
    # Interpolate missing months
    ts = ts.interpolate(method='linear')
    
    print(f"Time Series Length: {len(ts)} months")
    
    # Split Train/Test (80/20)
    train_size = int(len(ts) * 0.8)
    train_data = ts[:train_size]
    test_data = ts[train_size:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # SARIMA Parameters (from Notebook context)
    # Order: (4, 1, 1)
    # Seasonal Order: (1, 1, 1, 12)
    order = (4, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    print(f"\nFitting SARIMA{order}x{seasonal_order}")
    model = SARIMAX(train_data, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    sarima_fitted = model.fit(disp=False)
    print("Model Fitted.")
    print(sarima_fitted.summary())
    
    # Forecast
    print("\nGenerating Forecast")
    forecast_steps = len(test_data)
    forecast = sarima_fitted.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    # Evaluation Metrics
    mae = mean_absolute_error(test_data, forecast_values)
    mse = mean_squared_error(test_data, forecast_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_data, forecast_values)
    
    print("\nSARMIA Model Evaluation on Test Data:")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"  Mean Squared Error (MSE): {mse:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    plt.plot(test_data.index, test_data, label='Actual Test Data', color='green', marker='o', markersize=4)
    plt.plot(test_data.index, forecast_values, label='SARIMA Forecast', color='red', linestyle='--')
    
    # ARIMA Comparison (Adding this to provide context)
    print("\nFitting ARIMA(4, 1, 1) for comparison")
    from statsmodels.tsa.arima.model import ARIMA
    arima_model = ARIMA(train_data, order=(4, 1, 1))
    arima_fitted = arima_model.fit()
    
    arima_forecast = arima_fitted.get_forecast(steps=len(test_data))
    arima_pred = arima_forecast.predicted_mean
    
    arima_r2 = r2_score(test_data, arima_pred)
    arima_mae = mean_absolute_error(test_data, arima_pred)
    
    print(f"ARIMA R2: {arima_r2:.4f}")
    print(f"ARIMA MAE: {arima_mae:,.2f}")
    
    plt.plot(test_data.index, arima_pred, label=f'ARIMA Forecast (R2={arima_r2:.2f})', color='purple', linestyle=':')
    
    # Confidence Intervals for SARIMA
    plt.fill_between(test_data.index, 
                     conf_int.iloc[:, 0], 
                     conf_int.iloc[:, 1], 
                     color='red', alpha=0.1, 
                     label='95% Confidence Interval (SARIMA)')
                     
    plt.title('SARIMA vs ARIMA Model Forecast', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Mean Price (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_filename = 'model_comparison_plot.png'
    plt.savefig(plot_filename)
    print(f"\nComparison plot saved to {plot_filename}")

if __name__ == "__main__":
    analyze_sarima()
