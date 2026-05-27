import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

print("Starting Data Preparation for Power BI...")

# 1. Load Data (STRICT: No dummy data)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "train.csv")

if not os.path.exists(data_path):
    # Fallback to local 'data' if script_dir logic fails (unlikely)
    data_path = "data/train.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"CRITICAL ERROR: {data_path} not found. STOPPING.")

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
print(f"Dataset Loaded: {df.shape[0]:,} rows | Columns: {list(df.columns)}")

# 2. Filter: Store 1, Item 1
store_id, item_id = 1, 1
df_f = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()
df_f.set_index('date', inplace=True)
print(f"Filtered shape (Store {store_id}, Item {item_id}): {df_f.shape}")

# 3. Feature Engineering (sliding window approach)
lookback = 30
data = df_f['sales'].values.astype(float)

X, Y = [], []
for i in range(lookback, len(data)):
    # Features: last 30 days of sales + day of week + month
    window = data[i-lookback:i].tolist()
    dow = df_f.index[i].dayofweek
    month = df_f.index[i].month
    X.append(window + [dow, month])
    Y.append(data[i])

X, Y = np.array(X), np.array(Y)

# 4. Train/Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = Y[:train_size], Y[train_size:]

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples...")

# 5. Train GradientBoosting Model (No GPU, 100% stable on Mac)
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
print("Model training complete!")

# 6. Evaluate
test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, test_pred)
rmse = math.sqrt(mean_squared_error(y_test, test_pred))
print(f"\n--- Model Metrics ---")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 7. Future Forecast: Next 14 Days
print("Forecasting next 14 days...")
future_days = 14
future_preds = []
# Start with the last 30 days of real data
rolling_window = data[-lookback:].tolist()
last_date = df_f.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

for future_date in future_dates:
    dow = future_date.dayofweek
    month = future_date.month
    features = rolling_window[-lookback:] + [dow, month]
    pred = model.predict([features])[0]
    future_preds.append(round(pred, 0))
    rolling_window.append(pred)  # Feed prediction back for next step

# 8. Build Power BI Export DataFrame
# Historical (before test period)
test_start_idx = train_size + lookback
hist_dates = df_f.index[:test_start_idx]
df_hist = pd.DataFrame({
    'date': hist_dates,
    'actual_sales': df_f['sales'].iloc[:test_start_idx].values,
    'predicted_sales': np.nan,
    'forecast_sales': np.nan,
    'store': store_id, 'item': item_id,
    'data_category': 'Historical'
})

# Testing Phase
test_dates = df_f.index[test_start_idx:]
df_test = pd.DataFrame({
    'date': test_dates,
    'actual_sales': y_test,
    'predicted_sales': test_pred.round(0),
    'forecast_sales': np.nan,
    'store': store_id, 'item': item_id,
    'data_category': 'Testing (Model Evaluation)'
})

# Future Forecast
df_future = pd.DataFrame({
    'date': future_dates,
    'actual_sales': np.nan,
    'predicted_sales': np.nan,
    'forecast_sales': future_preds,
    'store': store_id, 'item': item_id,
    'data_category': 'Future Forecast'
})

# Combine & Export
final_df = pd.concat([df_hist, df_test, df_future], ignore_index=True)
export_dir = os.path.join(script_dir, "data")
os.makedirs(export_dir, exist_ok=True)
export_path = os.path.join(export_dir, "powerbi_export.csv")
final_df.to_csv(export_path, index=False)

print(f"\n✅ SUCCESS: {export_path} exported cleanly!")
print(f"   Total rows in export: {len(final_df):,}")
print(f"   MAE: {mae:.2f} | RMSE: {rmse:.2f}")
