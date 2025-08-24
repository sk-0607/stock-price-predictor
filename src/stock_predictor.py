#!/usr/bin/env python3
"""
Stock Price Predictor - ML Pipeline
A complete machine learning pipeline for predicting stock prices using historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def fetch_stock_data(symbol='AAPL', years=5):
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol (default: AAPL)
        years (int): Number of years of historical data (default: 5)
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    print(f"Fetching {years} years of {symbol} stock data...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    # Fetch data
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"No data found for {symbol}")
    
    print(f"Successfully fetched {len(data)} data points")
    return data

def perform_eda(data, symbol):
    """
    Perform Exploratory Data Analysis and create visualizations.
    
    Args:
        data (pd.DataFrame): Stock data
        symbol (str): Stock symbol for plot titles
    """
    print("Performing Exploratory Data Analysis...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Raw closing prices
    ax1.plot(data.index, data['Close'], label='Closing Price', color='blue', linewidth=1)
    ax1.set_title(f'{symbol} Stock Price - Raw Data')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving averages
    ax2.plot(data.index, data['Close'], label='Closing Price', color='blue', linewidth=1, alpha=0.7)
    ax2.plot(data.index, data['Close'].rolling(window=20).mean(), 
             label='20-day MA', color='red', linewidth=2)
    ax2.plot(data.index, data['Close'].rolling(window=50).mean(), 
             label='50-day MA', color='green', linewidth=2)
    ax2.set_title(f'{symbol} Stock Price - Moving Averages')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print basic statistics
    print(f"\nData Summary for {symbol}:")
    print(f"Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total Days: {len(data)}")
    print(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
    print(f"Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"Average Volume: {data['Volume'].mean():,.0f}")

def engineer_features(data):
    """
    Create features for machine learning model.
    
    Args:
        data (pd.DataFrame): Raw stock data
    
    Returns:
        pd.DataFrame: Data with engineered features
    """
    print("Engineering features...")
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Lag features (previous day's closing price)
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_2'] = df['Close'].shift(2)
    df['Close_Lag_3'] = df['Close'].shift(3)
    
    # Price changes
    df['Price_Change'] = df['Close'] - df['Close_Lag_1']
    df['Price_Change_Pct'] = df['Price_Change'] / df['Close_Lag_1']
    
    # Volatility (rolling standard deviation of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
    
    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    # Remove rows with NaN values (due to rolling calculations)
    df = df.dropna()
    
    print(f"Feature engineering complete. Final dataset shape: {df.shape}")
    return df

def prepare_target_and_features(df):
    """
    Prepare target variable and feature matrix.
    
    Args:
        df (pd.DataFrame): Data with engineered features
    
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    print("Preparing target and features...")
    
    # Target: next day's closing price
    df['Target'] = df['Close'].shift(-1)
    
    # Remove the last row (no target available)
    df = df[:-1]
    
    # Select features (exclude target and date-related columns)
    feature_columns = [col for col in df.columns if col not in 
                      ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    X = df[feature_columns]
    y = df['Target']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Features used: {feature_columns}")
    
    return X, y

def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple models.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
    
    Returns:
        dict: Dictionary containing models and their predictions
    """
    print("Training and evaluating models...")
    
    # Split data (last 20% as test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} Results:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R²: {r2:.4f}")
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'metrics': {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        }
    
    return results

def save_metrics(results, symbol):
    """
    Save model metrics to file.
    
    Args:
        results (dict): Model results dictionary
        symbol (str): Stock symbol
    """
    print("Saving metrics...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    with open('results/metrics.txt', 'w') as f:
        f.write(f"Stock Price Prediction Results for {symbol}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  RMSE: ${result['metrics']['RMSE']:.2f}\n")
            f.write(f"  MAE: ${result['metrics']['MAE']:.2f}\n")
            f.write(f"  R²: {result['metrics']['R2']:.4f}\n\n")
    
    print("Metrics saved to results/metrics.txt")

def create_visualization(results, symbol):
    """
    Create and save actual vs predicted plot.
    
    Args:
        results (dict): Model results dictionary
        symbol (str): Stock symbol
    """
    print("Creating visualization...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'red']
    
    for i, (model_name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot actual vs predicted
        ax.scatter(result['actual'], result['predictions'], 
                  alpha=0.6, color=colors[i], s=20)
        
        # Add perfect prediction line
        min_val = min(result['actual'].min(), result['predictions'].min())
        max_val = max(result['actual'].max(), result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title(f'{model_name} - Actual vs Predicted Prices')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f"RMSE: ${result['metrics']['RMSE']:.2f}\nMAE: ${result['metrics']['MAE']:.2f}\nR²: {result['metrics']['R2']:.4f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to results/actual_vs_predicted.png")
    plt.show()

def main():
    """
    Main function to run the complete stock prediction pipeline.
    """
    print("Stock Price Predictor")
    print("=" * 50)
    
    # Configuration
    symbol = 'AAPL'  # Default stock symbol
    years = 5        # Default years of data
    
    try:
        # Step 1: Fetch data
        data = fetch_stock_data(symbol, years)
        
        # Step 2: Perform EDA
        perform_eda(data, symbol)
        
        # Step 3: Feature engineering
        df_features = engineer_features(data)
        
        # Step 4: Prepare target and features
        X, y = prepare_target_and_features(df_features)
        
        # Step 5: Train and evaluate models
        results = train_and_evaluate_models(X, y)
        
        # Step 6: Save results
        save_metrics(results, symbol)
        create_visualization(results, symbol)
        
        print("\n Pipeline completed successfully!")
        print(f"Results saved in 'results/' directory")
        print(f"Check 'results/actual_vs_predicted.png' for visualizations")
        print(f"Check 'results/metrics.txt' for detailed metrics")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
