import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def check_and_load_data(filepath="aapl_data.csv"):
    """
    Checks if the stock data CSV exists. If not, runs generate_data.py.
    Then loads the CSV file.
    """
    if not os.path.exists(filepath):
        print(f"Data file '{filepath}' not found.")
        try:
            from generate_data import download_aapl_data
            download_aapl_data(filepath)
        except ImportError:
            print("Error: 'generate_data.py' helper script not found in the current directory.")
            sys.exit(1)
            
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def preprocess_and_feature_engineering(df):
    """
    Cleans the data and engineers lag and rolling features for time-series forecasting.
    """
    # Convert Date to datetime and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Store clean copy for reference
    processed_df = df.copy()
    
    # 1. Lag Features (using past Close prices to predict current Close)
    for lag in range(1, 6):
        processed_df[f'Close_Lag{lag}'] = processed_df['Close'].shift(lag)
        
    # 2. Rolling Window Features (based on past Close prices to prevent data leakage)
    # Since we want to predict Close of day t, our moving averages and volatility for day t 
    # must be computed using Close prices up to day t-1.
    processed_df['MA5'] = processed_df['Close'].shift(1).rolling(window=5).mean()
    processed_df['MA10'] = processed_df['Close'].shift(1).rolling(window=10).mean()
    processed_df['Volatility5'] = processed_df['Close'].shift(1).rolling(window=5).std()
    
    # Drop rows with missing values (due to lag and rolling calculations)
    # The first 10 rows will have NaN in MA10
    processed_df = processed_df.dropna().reset_index(drop=True)
    
    return processed_df

def train_and_evaluate(df):
    """
    Splits the data chronologically, trains a Linear Regression model,
    evaluates its performance, and returns the model and predictions.
    """
    # Define features and target
    features = ['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Close_Lag4', 'Close_Lag5', 'MA5', 'MA10', 'Volatility5']
    target = 'Close'
    
    X = df[features]
    y = df[target]
    
    # Chronological Split (80% Train, 20% Test) to mimic real-world forecasting
    train_size = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    dates_test = df['Date'].iloc[train_size:]
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("         EVALUATION METRICS")
    print("="*40)
    print(f"Mean Absolute Error (MAE):      ${mae:.4f}")
    print(f"Mean Squared Error (MSE):       ${mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):  ${rmse:.4f}")
    print(f"R-squared (R2 Score):           {r2:.4f}")
    print("="*40)
    
    # Print model parameters for university report details
    print("\nModel Intercept:", round(model.intercept_, 4))
    print("Model Coefficients:")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat}: {coef:.4f}")
        
    return model, X_test, y_test, y_pred, dates_test, features

def save_prediction_plot(dates, y_true, y_pred, filepath="aapl_prediction_plot.png"):
    """
    Creates a professional high-quality visualization comparing actual vs predicted stock prices.
    """
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Customize style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(dates, y_true.values, label='Actual AAPL Close', color='#1f77b4', linewidth=2, alpha=0.9)
    plt.plot(dates, y_pred, label='Predicted AAPL Close', color='#ff7f0e', linewidth=2, linestyle='--', alpha=0.9)
    
    plt.title('Apple Inc. (AAPL) Stock Price Prediction - Linear Regression', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Closing Price (USD)', fontsize=12)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(filepath)
    print(f"\nSaved prediction comparison plot to '{filepath}'.")
    plt.close()

def predict_next_day(df, model, features):
    """
    Predicts the closing price for the next trading day using the latest data.
    """
    # The last row in our processed dataframe
    last_row = df.iloc[-1]
    
    # Extract historical prices for next-day feature calculations
    # Note: we need the 10 most recent Close prices to calculate lag, moving averages, etc.
    # The most recent Close price is at index -1.
    recent_closes = df['Close'].iloc[-10:].values
    
    # Let's verify details:
    # Next day (T+1) features:
    # Close_Lag1 = Close(T)
    # Close_Lag2 = Close(T-1)
    # Close_Lag3 = Close(T-2)
    # Close_Lag4 = Close(T-3)
    # Close_Lag5 = Close(T-4)
    # MA5 = mean(Close(T) ... Close(T-4))
    # MA10 = mean(Close(T) ... Close(T-9))
    # Volatility5 = std(Close(T) ... Close(T-4))
    
    next_lag1 = recent_closes[-1]
    next_lag2 = recent_closes[-2]
    next_lag3 = recent_closes[-3]
    next_lag4 = recent_closes[-4]
    next_lag5 = recent_closes[-5]
    
    next_ma5 = np.mean(recent_closes[-5:])
    next_ma10 = np.mean(recent_closes[-10:])
    next_vol5 = np.std(recent_closes[-5:], ddof=1) # standard sample deviation
    
    next_features = pd.DataFrame([{
        'Close_Lag1': next_lag1,
        'Close_Lag2': next_lag2,
        'Close_Lag3': next_lag3,
        'Close_Lag4': next_lag4,
        'Close_Lag5': next_lag5,
        'MA5': next_ma5,
        'MA10': next_ma10,
        'Volatility5': next_vol5
    }])
    
    # Reorder columns to match feature list exactly
    next_features = next_features[features]
    
    # Predict next day Close
    predicted_close = model.predict(next_features)[0]
    
    # Calculate next trading day date
    last_date = pd.to_datetime(last_row['Date'])
    next_trading_day = last_date + pd.tseries.offsets.BDay(1)
    
    print("\n" + "="*40)
    print("      NEXT TRADING DAY PREDICTION")
    print("="*40)
    print(f"Last Available Date:     {last_date.strftime('%Y-%m-%d')} (Close: ${last_row['Close']:.2f})")
    print(f"Predicted Trading Date:  {next_trading_day.strftime('%Y-%m-%d')}")
    print(f"Predicted Closing Price: ${predicted_close:.2f}")
    print("="*40)
    
    return next_trading_day, predicted_close

def main():
    print("="*60)
    print("   AAPL Stock Closing Price Prediction Project - ML pipeline")
    print("="*60)
    
    # Step 1: Load data (will generate automatically if missing)
    df = check_and_load_data("aapl_data.csv")
    
    # Step 2: Feature engineering
    processed_df = preprocess_and_feature_engineering(df)
    print(f"Feature engineering completed. Active dataset size: {processed_df.shape[0]} rows after lag/rolling features.")
    
    # Step 3: Train model and evaluate
    model, X_test, y_test, y_pred, dates_test, features = train_and_evaluate(processed_df)
    
    # Step 4: Save visualization
    save_prediction_plot(dates_test, y_test, y_pred, "aapl_prediction_plot.png")
    
    # Step 5: Next-day forecast
    predict_next_day(processed_df, model, features)
    
    print("\nPipeline execution finished successfully.")

if __name__ == "__main__":
    main()
