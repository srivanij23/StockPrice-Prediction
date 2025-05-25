import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --- Load Data ---
@st.cache_data
def load_data():
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = '1981-01-01'
    df = yf.download('AAPL', start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = df.rename(columns=str.lower)
    df = df[['date', 'close', 'high', 'low', 'volume']].dropna()
    df['date'] = pd.to_datetime(df['date'])
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()

# --- Sidebar settings ---
st.sidebar.title("🔧 Settings")
target_col = st.sidebar.selectbox("Select target to predict:", ['close', 'high', 'low'])
look_back = 50

# --- Preprocessing ---
feature_cols = ['close', 'volume']
features = df[feature_cols].values
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_X.fit_transform(features)
target = df[[target_col]].values
target_scaled = scaler_y.fit_transform(target)

def create_dataset(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)

X, y = create_dataset(features_scaled, target_scaled, look_back)
dates = df['date'].values[look_back:]
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

st.write(f"📈 Total instances after preprocessing: {X.shape[0]}")
st.write(f"🧪 Training instances: {X_train.shape[0]}")
st.write(f"🧾 Testing instances: {X_test.shape[0]}")

# --- Train LSTM ---
@st.cache_resource
def train_lstm_model(X_train, y_train, look_back):
    model = Sequential([
        LSTM(64, input_shape=(look_back, 2)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    return model

model = train_lstm_model(X_train, y_train, look_back)

# --- Predict ---
lstm_pred = model.predict(X_test)
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
lstm_pred_rescaled = scaler_y.inverse_transform(lstm_pred)

# --- XGBoost Residual Learning ---
residuals = y_test_rescaled - lstm_pred_rescaled
xgb_input = X_test.reshape(X_test.shape[0], -1)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(xgb_input, residuals.ravel())
xgb_pred = xgb_model.predict(xgb_input).reshape(-1, 1)
hybrid_pred = lstm_pred_rescaled + xgb_pred

# --- Compute and Cache RMSE ---
@st.cache_data
def compute_rmse(y_test_rescaled, lstm_pred_rescaled, hybrid_pred):
    lstm_rmse = np.sqrt(mean_squared_error(y_test_rescaled, lstm_pred_rescaled))
    hybrid_rmse = np.sqrt(mean_squared_error(y_test_rescaled, hybrid_pred))
    return lstm_rmse, hybrid_rmse

lstm_rmse, hybrid_rmse = compute_rmse(y_test_rescaled, lstm_pred_rescaled, hybrid_pred)

# --- Compute Custom Accuracy ---
mean_actual = np.mean(y_test_rescaled)
lstm_accuracy = 100 - (lstm_rmse / mean_actual) * 100
hybrid_accuracy = 100 - (hybrid_rmse / mean_actual) * 100

# --- Evaluation ---
st.subheader(f"📊 Model Evaluation: {target_col.upper()} Prediction")
st.write(f"LSTM RMSE: {lstm_rmse:.4f} | ✅ Accuracy: {lstm_accuracy:.2f}%")
st.write(f"Hybrid RMSE: {hybrid_rmse:.4f} | ✅ Accuracy: {hybrid_accuracy:.2f}%")

# --- Select Date ---
date_list = [pd.to_datetime(str(d)).date() for d in dates_test]
selected_date = st.selectbox("📅 Select Date to Inspect: ", date_list)
idx = date_list.index(selected_date)

# --- Show Predictions for Selected Date ---
st.write(f"### 📌 {selected_date}")
st.write(f"🔵 Actual {target_col}: ${y_test_rescaled[idx][0]:.2f}")
st.write(f"🟠 LSTM Predicted: ${lstm_pred_rescaled[idx][0]:.2f}")
st.write(f"🟢 Hybrid Predicted: ${hybrid_pred[idx][0]:.2f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test_rescaled, label='🔵 Actual', color='blue')
ax.plot(lstm_pred_rescaled, label='🟠 LSTM')
ax.plot(hybrid_pred, label='🟢 Hybrid')
ax.axvline(x=idx, color='gray', linestyle='--', label='📍 Selected Date')
ax.set_title(f"Model Predictions ({target_col.capitalize()})")
ax.set_xlabel("Time Index")
ax.set_ylabel(target_col.capitalize())
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- 30-Day Forecast (Skipping Weekends) ---
st.write("\n## 🔮 30-Day Forecast (Weekdays Only)")

X_future = features_scaled[-look_back:].tolist()
future_preds = []
future_dates = []
last_date = df['date'].iloc[-1].date()

while len(future_preds) < 30:
    last_date += pd.Timedelta(days=1)
    if last_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        continue

    seq_input = np.array(X_future[-look_back:])
    lstm_next_scaled = model.predict(seq_input.reshape(1, look_back, 2))
    lstm_next = scaler_y.inverse_transform(lstm_next_scaled)[0][0]
    xgb_input_future = seq_input.reshape(1, -1)
    xgb_residual = xgb_model.predict(xgb_input_future)[0]
    hybrid_next = lstm_next + xgb_residual

    future_preds.append((lstm_next, hybrid_next))
    future_dates.append(last_date)

    last_volume = scaler_X.inverse_transform([X_future[-1]])[0][1]
    next_input = scaler_X.transform([[lstm_next, last_volume]])
    X_future.append(next_input[0])

# Show forecast results
st.write("### 🗓 Forecasted Dates and Values")
for date, preds in zip(future_dates, future_preds):
    lstm_val, hybrid_val = preds
    st.write(f"{date}: 🟠 LSTM = ${lstm_val:.2f}, 🟢 Hybrid = ${hybrid_val:.2f}")