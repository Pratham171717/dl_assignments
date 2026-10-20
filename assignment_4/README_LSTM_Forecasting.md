# LSTM-Based Stock Price Forecasting

An LSTM model that forecasts Apple's (AAPL) daily closing stock price from the previous 60 days of prices.

## Problem Statement
Develop an LSTM-based model for time-series forecasting using stock price, weather, or sales datasets.

## Dataset
- **Source:** Apple stock prices CSV from the [plotly datasets repository](https://github.com/plotly/datasets) (`finance-charts-apple.csv`), loaded by URL
- **Size:** 506 trading days (February 2015 to February 2017), no missing values
- **Feature used:** `AAPL.Close` (closing price), scaled to 0–1 with `MinMaxScaler`
- **Split:** 80% training (404 days) / 20% testing (102 days), in time order
- **Sequences:** a sliding window of 60 days predicts the next day's close

## Model Architecture
`LSTM(50, return_sequences=True)` → `Dropout(0.2)` → `LSTM(50)` → `Dropout(0.2)` → `Dense(25, ReLU)` → `Dense(1)`

Total parameters: 31,901. Training: Adam optimizer, MSE loss, batch size 32, 5 epochs, 10% validation split.

## Requirements
- Python 3, TensorFlow / Keras
- NumPy, pandas, matplotlib, scikit-learn
- Google Colab (runs in under a minute, no GPU needed)

## How to Run
1. Open `LSTM.ipynb` in Google Colab.
2. Click **Runtime → Run all**.

## Results (test set, 102 days)
| Metric | Value |
|---|---|
| MSE | 27.3506 |
| RMSE | 5.2298 |
| MAE | 4.3089 |

MAE is about 3.8% of the average closing price (about $113). The model's forecast for the next closing price was **125.02**.

## Observations
- The predicted curve follows the overall trend but is smoother and lags behind the actual prices.
- It underestimates the sharp rise at the end of the test period, because the test prices are higher than most prices seen in training.
- Validation loss was lowest at epoch 3; the validation set is small (35 sequences), so changes after that are noisy.

## Possible Improvements
- Train for more epochs on a longer price history.
- Add more features (volume, moving averages) and tune the window size.
- Fit the scaler on the training data only, to avoid using test-period information.

## Files
- `LSTM.ipynb`: notebook with code and outputs
- `Assignment_LSTM_Stock_Forecasting.docx`: assignment report
