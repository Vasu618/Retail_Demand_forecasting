
## GOAL
# Predict future product demand (sales) using historical time-series data to help businesses optimize inventory, reduce stockouts, and improve decision-making.


# ##  1. DATA UNDERSTANDING



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

# Load the dataset
import pandas as pd
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
    
# Display basic info
print("\n--- Dataset Info ---")
df.info()

# Display summary statistics
print("\n--- Summary Statistics ---")
print(df.describe())




# ##  2. DATA PREPROCESSING

# %%
# 1. Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# 2. Sort data chronologically
df = df.sort_values('date')

# 3. Select a specific store and item (store=1, item=1)
# Focusing on one pair makes the problem a univariate time-series task initially.
store_id = 1
item_id = 1
df_filtered = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()

print(f"Filtered data shape for Store {store_id}, Item {item_id}: {df_filtered.shape}")

# Set date as index
df_filtered.set_index('date', inplace=True)

# Extract only the target variable
ts_data = df_filtered[['sales']]

# Handle missing values using interpolation (if any exist)
if ts_data.isnull().sum().values[0] > 0:
    print("Handling missing values via time interpolation...")
    ts_data = ts_data.interpolate(method='time')

# 4. Apply scaling using MinMaxScaler
# LSTMs are sensitive to the scale of the input data, so we scale values between 0 and 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(ts_data)

print(f"Scaled data shape: {scaled_data.shape}")


#  3. TRAIN-TEST SPLIT

train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size

# Split the data sequentially
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")


#  4. SEQUENCE CREATION (CRITICAL STEP)



def create_sequences(dataset, lookback=30):
    """
    Converts time-series data into sequences for LSTM.
    Uses a sliding window approach.
    """
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        # Input sequence: past 'lookback' days
        X.append(dataset[i:(i + lookback), 0])
        # Target output: the value on the next day
        Y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(Y)

# Define window size
lookback = 30 

# Create sequences for training and testing
X_train, y_train = create_sequences(train_data, lookback)
X_test, y_test = create_sequences(test_data, lookback)

# Reshape input to be [samples, time steps, features]
# We have 1 feature (sales)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"X_train reshaped: {X_train.shape} -> (samples={X_train.shape[0]}, time_steps={X_train.shape[1]}, features={X_train.shape[2]})")
print(f"y_train shape: {y_train.shape}")
print(f"X_test reshaped: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# %% [markdown]
# ##  5. MODEL BUILDING (LSTM)



model = Sequential()

# First LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Dense output layer
model.add(Dense(units=1))

# Compile the model
# Adam optimizer is great for deep learning, MSE is standard for regression
model.compile(optimizer='adam', loss='mean_squared_error')

# Show model architecture
model.summary()


# 6. TRAINING
# Train the model over a set number of epochs. We use validation data to monitor how well the model generalizes during training.


epochs = 20  
batch_size = 32

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=1
)

# Plot training & validation loss curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss (Mean Squared Error)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


#  7. EVALUATION
# 

# %%
# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])

test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])

# Calculate error metrics for test data
test_rmse = math.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
test_mae = mean_absolute_error(y_test_inv[0], test_predict[:,0])

print(f"\n--- Evaluation Metrics ---")
print(f"Test RMSE: {test_rmse:.2f} (Root Mean Squared Error)")
print(f"Test MAE:  {test_mae:.2f} (Mean Absolute Error)")


# ##  8. VISUALIZATION
# 
# Visualizing the predictions mapped over the actual historical timeline.

# %%
# Prepare data for plotting to align with the original time axis
# Shift train predictions for plotting
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[lookback:len(train_predict)+lookback, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(lookback*2):len(scaled_data), :] = test_predict

# Plot baseline and predictions (Full View)
plt.figure(figsize=(15, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Sales', color='lightgray')
plt.plot(train_predict_plot, label='Train Predictions', color='blue', alpha=0.7)
plt.plot(test_predict_plot, label='Test Predictions', color='red', alpha=0.7)
plt.title(f'Demand Forecasting for Store {store_id}, Item {item_id}')
plt.xlabel('Time (Days)')
plt.ylabel('Sales Volume')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Zoomed-in view on a portion of the test set
plt.figure(figsize=(15, 6))
zoom_start = len(train_predict) + (lookback*2)
zoom_end = zoom_start + 150 # Look at the first 150 days of testing
plt.plot(scaler.inverse_transform(scaled_data)[zoom_start:zoom_end], label='Actual Sales', color='black', marker='.')
plt.plot(test_predict_plot[zoom_start:zoom_end], label='Predicted Sales', color='red', marker='x')
plt.title('Zoomed-in View: Actual vs Predicted (150 Days)')
plt.xlabel('Time (Days)')
plt.ylabel('Sales Volume')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()




# ##  12. FUTURE FORECASTING
# 
# Now we will use the trained model to predict the next 14 days of demand into the unknown future!
# To do this, we take the last known 30 days of data, predict day 31. Then we append the prediction to our data, slide the window, and predict day 32, and so on.


future_days = 14
# Get the last 'lookback' days from the scaled dataset
last_30_days = scaled_data[-lookback:]

# We need to iteratively predict
current_sequence = last_30_days.reshape((1, lookback, 1))
future_predictions_scaled = []

print(f"\nPredicting the next {future_days} days...")
for _ in range(future_days):
    # Predict the next day
    next_day_pred = model.predict(current_sequence, verbose=0)
    future_predictions_scaled.append(next_day_pred[0, 0])
    
    # Update the sequence: remove the first day, append the predicted day
    # current_sequence is shape (1, 30, 1). next_day_pred is (1, 1)
    next_day_reshaped = np.reshape(next_day_pred, (1, 1, 1))
    current_sequence = np.append(current_sequence[:, 1:, :], next_day_reshaped, axis=1)

# Inverse transform the predictions back to real sales numbers
future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

# Create future dates for plotting
last_date = df_filtered.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

print("\n--- FUTURE DEMAND FORECAST (Next 14 Days) ---")
for date, pred in zip(future_dates, future_predictions):
    print(f"{date.strftime('%Y-%m-%d')}: {int(round(pred[0]))} units")

# Plot the future forecast
plt.figure(figsize=(10, 5))
# Plot the last 60 days of actual data for context
plt.plot(df_filtered.index[-60:], df_filtered['sales'].values[-60:], label='Recent Actual Sales', color='black', marker='.')
plt.plot(future_dates, future_predictions, label='Future Forecast (14 days)', color='green', marker='o', linewidth=2)
plt.title(' 14-Day Future Demand Forecast')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
