import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style for professional plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (14, 7), 'figure.dpi': 300, 'font.size': 12})

# 1. Create 'images' folder
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
print(f"Directory '{output_dir}/' is ready.")

# 2. Load and validate data
data_path = "data/train.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"CRITICAL ERROR: {data_path} not found. Please provide the real dataset.")

print("Loading dataset...")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
print("Dataset loaded successfully.")

# 3. Filter data
store_id = 1
item_id = 1
df_filtered = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()
df_filtered.set_index('date', inplace=True)

# ---------------------------------------------------------
# VISUALIZATION 1: Time-Series Plot (Demand Forecasting)
# ---------------------------------------------------------
print("Generating Demand_forecasting.png...")
plt.figure()
plt.plot(df_filtered.index, df_filtered['sales'], color='steelblue', linewidth=1, alpha=0.7, label='Daily Sales')
# Add a 30-day moving average for the trend
plt.plot(df_filtered.index, df_filtered['sales'].rolling(window=30).mean(), color='darkred', linewidth=2, label='30-Day Trend (Moving Avg)')
plt.title(f'Store {store_id}, Item {item_id}: Full Sales History Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Units Sold', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Demand_forecasting.png'))
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 2: Zoomed View
# ---------------------------------------------------------
print("Generating Zoomed_in_view.png...")
plt.figure()
last_90 = df_filtered.iloc[-90:]
plt.plot(last_90.index, last_90['sales'], marker='o', linestyle='-', color='teal', linewidth=2)
plt.title('Zoomed In View: Last 90 Days of Sales Activity', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Units Sold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Zoomed_in_view.png'))
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 3: Train vs Test Split Visualization
# ---------------------------------------------------------
print("Generating Train_vs_Test.png...")
plt.figure()
train_size = int(len(df_filtered) * 0.8)
train_split = df_filtered.iloc[:train_size]
test_split = df_filtered.iloc[train_size:]
plt.plot(train_split.index, train_split['sales'], label='Training Data (80%)', color='royalblue', alpha=0.8)
plt.plot(test_split.index, test_split['sales'], label='Testing Data (20%)', color='darkorange', alpha=0.8)
plt.axvline(x=train_split.index[-1], color='black', linestyle='--', linewidth=2, label='Train/Test Split Point')
plt.title('Chronological Data Split: Train vs. Test Sets', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Units Sold', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Train_vs_Test.png'))
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 7: Multi-Series Comparison
# ---------------------------------------------------------
print("Generating Multi-Series_Comparison.png...")
plt.figure()
df_s1 = df[(df['store'] == 1) & (df['item'] == 1)].set_index('date')
df_s2 = df[(df['store'] == 2) & (df['item'] == 1)].set_index('date')
plt.plot(df_s1.index, df_s1['sales'].rolling(30).mean(), label='Store 1 - Item 1 (Trend)', color='purple', linewidth=2)
plt.plot(df_s2.index, df_s2['sales'].rolling(30).mean(), label='Store 2 - Item 1 (Trend)', color='green', linewidth=2)
plt.title('Multi-Series Comparison: Sales Trends Across Different Stores', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Units Sold (30-Day Moving Avg)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Multi-Series_Comparison.png'))
plt.close()


# ---------------------------------------------------------
# PREPARE MODEL FOR PREDICTIONS & LOSS
# ---------------------------------------------------------
print("Training model to generate Actual vs Predicted and Loss curves...")
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df_filtered[['sales']])

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_seq(data, lookback=30):
    X, Y = [], []
    for i in range(len(data)-lookback):
        X.append(data[i:i+lookback, 0])
        Y.append(data[i+lookback, 0])
    return np.array(X), np.array(Y)

lookback = 30
X_train, y_train = create_seq(train_data, lookback)
X_test, y_test = create_seq(test_data, lookback)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model and capture history
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# ---------------------------------------------------------
# VISUALIZATION 5: Model Loss Curve
# ---------------------------------------------------------
print("Generating model_loss.png...")
plt.figure()
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', marker='o', linewidth=2)
plt.title('Model Optimization: Training vs Validation Loss', fontsize=16, fontweight='bold')
plt.xlabel('Training Epochs', fontsize=14)
plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_loss.png'))
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 4: Actual vs Predicted Plot
# ---------------------------------------------------------
print("Generating Actual_vs_predicted.png...")
test_pred = model.predict(X_test, verbose=0)
test_pred_inv = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure()
test_dates = df_filtered.index[train_size+lookback:]
plt.plot(test_dates, y_test_inv, label='Actual Sales (Ground Truth)', color='gray', alpha=0.6, linewidth=2)
plt.plot(test_dates, test_pred_inv, label='Predicted Sales (LSTM)', color='crimson', alpha=0.9, linewidth=2)
plt.title('Actual vs Predicted Sales on Test Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Units Sold', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Actual_vs_predicted.png'))
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 6: 14-Day Future Forecast
# ---------------------------------------------------------
print("Generating 14-Day-future-demand.png...")
future_days = 14
last_seq = scaled_data[-lookback:].reshape((1, lookback, 1))
future_preds = []

for _ in range(future_days):
    pred = model.predict(last_seq, verbose=0)
    future_preds.append(pred[0,0])
    last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
last_date = df_filtered.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

plt.figure()
# Plot the last 45 days for better context ratio
plt.plot(df_filtered.index[-45:], df_filtered['sales'].iloc[-45:], label='Recent Actual Sales', color='black', marker='.', linewidth=2)
plt.plot(future_dates, future_preds_inv, label='Future Forecast (14 Days)', color='forestgreen', marker='o', linewidth=2, markersize=8)
plt.title('Forward Looking: 14-Day Future Demand Forecast', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Projected Units Sold', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '14-Day-future-demand.png'))
plt.close()

print("\nSUCCESS: All visualizations generated and saved in the 'images/' folder!")
