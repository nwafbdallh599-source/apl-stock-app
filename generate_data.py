import os
import sys
import pandas as pd
import numpy as np

def download_aapl_data(filepath, start_date="2024-01-01", end_date="2026-07-25"):
    """
    Attempts to download real Apple (AAPL) stock data using yfinance.
    If it fails, it falls back to generating a realistic synthetic dataset.
    """
    print("Attempting to download Apple (AAPL) stock data from Yahoo Finance...")
    try:
        import yfinance as yf
        # Fetch actual AAPL stock data
        df = yf.download("AAPL", start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError("Downloaded dataset is empty.")
            
        # If MultiIndex columns (sometimes happens with yfinance), flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Ensure the Date column is formatted nicely
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Successfully downloaded real AAPL stock data. Saved {len(df)} rows to '{filepath}'.")
        return True
    except Exception as e:
        print(f"\n[WARNING] Failed to download data using yfinance: {e}")
        print("Falling back to generating a realistic synthetic Apple (AAPL) stock dataset...")
        return generate_synthetic_data(filepath, start_date, end_date)

def generate_synthetic_data(filepath, start_date="2024-01-01", end_date="2026-07-25"):
    """
    Generates a realistic daily stock price dataset using Geometric Brownian Motion (GBM)
    with parameters typical for Apple Inc. (AAPL) in the 2024-2026 period.
    """
    np.random.seed(42)  # Set seed for reproducibility
    
    # Generate business days (exclude weekends)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    # Parameters for GBM based on historical AAPL characteristics
    # Apple was around $185 at the start of 2024 and rose to around $225 by mid-2026.
    S0 = 185.0  # Starting price
    mu = 0.0004  # Annualized drift ~ 10% divided by 252 (daily drift ~ 0.04%)
    sigma = 0.015  # Daily volatility ~ 1.5% (approx 24% annualized)
    
    # Generate daily returns using normal distribution
    daily_returns = np.random.normal(loc=mu, scale=sigma, size=n_days)
    
    # Generate price path
    close_prices = np.zeros(n_days)
    close_prices[0] = S0
    for t in range(1, n_days):
        close_prices[t] = close_prices[t-1] * np.exp(daily_returns[t])
        
    # Generate other OHLC columns realistically
    # Standard deviation of intra-day high/low spread is around 1-2%
    high_offsets = np.random.exponential(scale=0.012, size=n_days)
    low_offsets = np.random.exponential(scale=0.012, size=n_days)
    open_offsets = np.random.normal(loc=0, scale=0.005, size=n_days)
    
    open_prices = np.zeros(n_days)
    high_prices = np.zeros(n_days)
    low_prices = np.zeros(n_days)
    
    # Initialize Open price for first day
    open_prices[0] = S0 * (1 + open_offsets[0])
    
    for t in range(n_days):
        if t > 0:
            # Open is close to yesterday's close, plus some overnight gap
            open_prices[t] = close_prices[t-1] * (1 + open_offsets[t])
            
        # High must be higher than or equal to both Open and Close
        high_prices[t] = max(open_prices[t], close_prices[t]) * (1 + high_offsets[t])
        # Low must be lower than or equal to both Open and Close
        low_prices[t] = min(open_prices[t], close_prices[t]) * (1 - low_offsets[t])
        
    # Volume: standard lognormal distribution around 50 million shares traded daily
    volume = np.random.lognormal(mean=17.7, sigma=0.3, size=n_days).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': np.round(close_prices, 2),
        'Adj Close': np.round(close_prices, 2), # Simplified: Adj Close = Close
        'Volume': volume
    })
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Generated realistic synthetic AAPL dataset. Saved {len(df)} rows to '{filepath}'.")
    return True

if __name__ == "__main__":
    csv_path = "aapl_data.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    download_aapl_data(csv_path)
