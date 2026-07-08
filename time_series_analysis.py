
"""
Time Series Analysis - Airline Passengers
----------------------------------------
This script performs:
1. Data Loading
2. Time Series Visualization
3. Quarterly Resampling
4. Seasonal Decomposition
5. Moving Average Analysis
6. SARIMA Forecasting
7. RMSE Evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(
    "data.csv",           # Change filename if required
    parse_dates=["Month"],
    index_col="Month"
)

print("=" * 50)
print("First 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())


# -----------------------------
# 2. Original Time Series
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(df["Passengers"], marker="o")
plt.title("Monthly Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.grid(True)
plt.show()


# -----------------------------
# 3. Quarterly Resampling
# -----------------------------
quarterly = df.resample("Q").mean()

print("\nQuarterly Data")
print(quarterly.head())

plt.figure(figsize=(12,5))
plt.plot(quarterly.index, quarterly["Passengers"], marker="o")
plt.title("Quarterly Average Passengers")
plt.grid(True)
plt.show()


# -----------------------------
# 4. Seasonal Decomposition
# -----------------------------
decomposition = seasonal_decompose(
    df["Passengers"],
    model="multiplicative",
    period=12
)

fig = decomposition.plot()
fig.set_size_inches(12,8)
plt.show()


# -----------------------------
# 5. Moving Averages
# -----------------------------
df["MA_6"] = df["Passengers"].rolling(window=6).mean()
df["MA_12"] = df["Passengers"].rolling(window=12).mean()

plt.figure(figsize=(12,6))
plt.plot(df["Passengers"], label="Actual")
plt.plot(df["MA_6"], "--", label="6-Month MA")
plt.plot(df["MA_12"], color="red", label="12-Month MA")
plt.title("Moving Average Smoothing")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 6. Train/Test Split
# -----------------------------
train = df["Passengers"].iloc[:-12]
test = df["Passengers"].iloc[-12:]

print("\nTraining Samples :", len(train))
print("Testing Samples  :", len(test))


# -----------------------------
# 7. SARIMA Model
# -----------------------------
model = SARIMAX(
    train,
    order=(2,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False)

forecast = result.forecast(steps=12)


# -----------------------------
# 8. Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(test, forecast))

print("\nRMSE =", round(rmse,2), "passengers")


# -----------------------------
# 9. Forecast Plot
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual", color="blue")
plt.plot(test.index, forecast, "--", color="red", label="Forecast")

plt.fill_between(
    test.index,
    forecast * 0.8,
    forecast * 1.2,
    alpha=0.2
)

plt.title(f"SARIMA Forecast (RMSE={rmse:.2f})")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 10. Forecast Table
# -----------------------------
forecast_df = pd.DataFrame({
    "Actual": test,
    "Forecast": forecast
})

print("\nForecast Comparison")
print(forecast_df)


# -----------------------------
# 11. Observations
# -----------------------------
print("\nSuggested Report Observations")
print("- Passenger count shows a clear upward trend.")
print("- Strong yearly seasonality exists.")
print("- 12-month moving average captures long-term trend.")
print("- Quarterly resampling smooths monthly fluctuations.")
print("- SARIMA effectively models trend and seasonality.")
print(f"- Forecast RMSE = {rmse:.2f} passengers.")
