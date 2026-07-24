# Apple Inc. (AAPL) Stock Price Prediction

A complete, university-grade machine learning project in Python that utilizes **Linear Regression** to predict the daily closing price of Apple Inc. (AAPL). 

The project contains a complete pipeline, including data downloading (from Yahoo Finance), data cleaning, feature engineering (lags, rolling averages, volatility), model training, chronological train-test splitting, model evaluation, visualization, and a next-day stock price prediction.

---

## Project Structure

```text
aapl_stock_prediction/
│
├── aapl_data.csv            # Cleaned historical stock prices (auto-generated)
├── generate_data.py         # Script to fetch real AAPL stock data or generate simulated data
├── main.py                  # Main ML pipeline (load, engineer, train, test, evaluate, plot, predict)
├── aapl_prediction_plot.png # Graph comparing actual vs. predicted closing prices on the test set
├── requirements.txt         # Project package dependencies
├── README.md                # General project overview and setup guidelines (this file)
└── Report.md                # Comprehensive academic report for university submission
```

---

## Features Built
To avoid the common mistake of **data leakage** (e.g., using same-day high/low/open prices to predict same-day close, which is mathematically trivial and useless for trading), this project uses **historical lag and rolling features**:
* **Lag Features (`Close_Lag1` to `Close_Lag5`)**: Closing prices of the preceding 1 to 5 trading days.
* **Simple Moving Average 5-day (`MA5`)**: 5-day moving average of closing prices up to yesterday.
* **Simple Moving Average 10-day (`MA10`)**: 10-day moving average of closing prices up to yesterday.
* **Rolling Volatility (`Volatility5`)**: 5-day sample standard deviation of closing prices up to yesterday.

---

## Setup & Installation

### Prerequisites
Make sure you have Python 3.8+ installed. 

### 1. Clone or Copy the Project
Ensure all project files are located in the same directory:
`C:\Users\نواف الريبة\.gemini\antigravity\scratch\aapl_stock_prediction\`

### 2. Install Dependencies
Install all required libraries via `pip` (or `py -m pip` on Windows):
```bash
pip install -r requirements.txt
```
The dependencies include:
* `pandas` - For data manipulation and analysis.
* `numpy` - For numerical calculations.
* `matplotlib` - For plotting high-quality graphs.
* `scikit-learn` - For training and evaluating the Linear Regression model.
* `yfinance` - For downloading actual real-time stock data.

---

## How to Run

Execute the main pipeline using Python:
```bash
python main.py
```
*(On Windows, you can also use `py main.py`)*

### Program Output flow:
1. Checks for `aapl_data.csv`. If missing, it uses `yfinance` to download Apple stock data from Jan 2024 to the present. If offline, it automatically falls back to generating a realistic synthetic dataset.
2. Cleans the data and constructs lag and rolling features.
3. Splits the data chronologically (80% training set, 20% test set).
4. Trains the `LinearRegression` model.
5. Prints evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (\(R^2\)).
6. Generates and saves a high-quality visualization named `aapl_prediction_plot.png`.
7. Outputs the predicted closing price for the next trading day.

---

## Performance Summary

When evaluated on the test set of actual historical data:
* **Mean Absolute Error (MAE)**: \$3.68 (The model's predictions deviate by an average of \$3.68 from the actual price).
* **Root Mean Squared Error (RMSE)**: \$4.94.
* **R-squared (\(R^2\))**: 0.9573 (The engineered features explain **95.73%** of the variance in Apple's closing stock price).

---

## Next-Day Prediction Example
* **Last Available Date**: 2026-07-24 (Actual Close: \$332.75)
* **Predicted Trading Date**: 2026-07-27 (Monday)
* **Predicted Closing Price**: **\$332.71**
