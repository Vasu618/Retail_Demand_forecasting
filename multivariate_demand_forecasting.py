# %% [markdown]
# # 📌 Advanced Multivariate Retail Demand Forecasting
# 
# ## 🎯 GOAL
# Improve our baseline LSTM model by adding **time-based features** (Day of Week, Month). This helps the model explicitly learn weekly and yearly seasonal patterns, significantly improving its predictive power over using just historical sales alone.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. LOAD DATA & FEATURE ENGINEERING
# We will extract cyclical time features directly from the 'date' column.

# %%
# Load data
import os

if not os.path.exists("data/train.csv"):
    raise FileNotFoundError("CRITICAL ERROR: data/train.csv not found. Please ensure the actual dataset is placed in the 'data' folder. STOPPING execution.")

df = pd.read_csv("data/train.csv")
print("Dataset Loaded Successfully")
print(df.head())

# DATA VALIDATION STEP
print("\n--- Data Validation ---")
print(f"Dataset shape: {df.shape}")

required_columns = ['date', 'store', 'item', 'sales']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"CRITICAL ERROR: Missing required columns in the dataset: {missing_columns}. STOPPING execution.")
else:
    print("All required columns (date, store, item, sales) are present.")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Filter for one store/item for demonstration
df_filtered = df[(df['store'] == 1) & (df['item'] == 1)].copy()
df_filtered.set_index('date', inplace=True)

# --- NEW MULTIVARIATE FEATURE ENGINEERING ---
# Extracting properties from the date
df_filtered['day_of_week'] = df_filtered.index.dayofweek  # 0=Monday, 6=Sunday
df_filtered['month'] = df_filtered.index.month          # 1 to 12

# Our inputs now include sales AND time features
features = ['sales', 'day_of_week', 'month']
ts_data = df_filtered[features]

print("\n--- New Multivariate Dataset Head ---")
print(ts_data.head())

# %% [markdown]
# ## 2. SCALING MULTIVARIATE DATA
# We scale all features to [0,1]. We will save a scaler for the target variable separately so we can easily un-scale our predictions back to actual sales numbers later.

# %%
# Scale all features (sales, day_of_week, month)
scaler_all = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler_all.fit_transform(ts_data)

# Create a separate scaler specifically for the 'sales' column (which is at index 0)
# This prevents errors when trying to inverse-transform the single-dimension prediction.
scaler_target = MinMaxScaler(feature_range=(0, 1))
scaler_target.fit(ts_data[['sales']])

# %% [markdown]
# ## 3. TRAIN-TEST SPLIT & MULTIVARIATE SEQUENCES

# %%
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

def create_multivariate_sequences(dataset, lookback=30):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        # X gets ALL features for the lookback period
        X.append(dataset[i:(i + lookback), :])
        # Y gets ONLY the target variable (sales, index 0) for the next day
        Y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(Y)

lookback = 30
X_train, y_train = create_multivariate_sequences(train_data, lookback)
X_test, y_test = create_multivariate_sequences(test_data, lookback)

print(f"\nX_train shape: {X_train.shape} -> (samples, time_steps, features)")
print(f"Number of features used: {X_train.shape[2]} (sales, day_of_week, month)")

# %% [markdown]
# ## 4. BUILD & TRAIN MULTIVARIATE LSTM
# Notice how `input_shape` automatically picks up the new features dimension (3 instead of 1).

# %%
model = Sequential()
# Input shape takes (time_steps, number_of_features)
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# %% [markdown]
# ## 5. EVALUATION
# Because we used a dedicated scaler for the target variable, un-scaling our 1D predictions is straightforward.

# %%
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions back to original sales scale
train_predict = scaler_target.inverse_transform(train_predict)
y_train_inv = scaler_target.inverse_transform([y_train])

test_predict = scaler_target.inverse_transform(test_predict)
y_test_inv = scaler_target.inverse_transform([y_test])

test_rmse = math.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
test_mae = mean_absolute_error(y_test_inv[0], test_predict[:,0])

print(f"\n--- Multivariate Evaluation Metrics ---")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE:  {test_mae:.2f}")

# %% [markdown]
# ## 6. VISUALIZATION

# %%
plt.figure(figsize=(15, 6))
# Only plot the sales portion of the actual data
plt.plot(df_filtered.index[lookback:], df_filtered['sales'].values[lookback:], label='Actual Sales', color='lightgray')

# Generate proper indices for plotting to match dates correctly
train_idx = df_filtered.index[lookback:lookback+len(train_predict)]
test_idx = df_filtered.index[lookback+len(train_predict)+lookback : lookback+len(train_predict)+lookback+len(test_predict)]

plt.plot(train_idx, train_predict, label='Train Predictions', color='blue', alpha=0.7)
plt.plot(test_idx, test_predict, label='Test Predictions', color='red', alpha=0.7)

plt.title('Advanced Multivariate Demand Forecasting (Sales + Time Features)')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
