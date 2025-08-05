import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Page setup ....
st.set_page_config(page_title="Hourly Energy Consumption Forecast", layout="wide")
st.image("Energy Forecasting in a Modern Office.png", use_container_width=True, caption="Hyper-Realistic Forecasting Visualization")
st.title("⚡ Hourly Energy Consumption Forecast")
st.markdown("Forecast hourly energy usage for the next 1 to 30 days using a trained LSTM model.")

# Sidebar...
forecast_hours = st.sidebar.selectbox("Select Forecast Duration:", [24, 72, 168, 720],
                                      format_func=lambda x: f"{x} hours ({x//24} days)")

# Load data...
data = pd.read_csv("PJMW_MW_Hourly.csv", parse_dates=["Datetime"])
data = data.sort_values("Datetime")
data.set_index("Datetime", inplace=True)

# Show recent data.....
st.subheader("📊 Recent Energy Consumption")
st.line_chart(data["PJMW_MW"].tail(168))

# Scale and prepare sequence....
scaler = MinMaxScaler()
data["Scaled"] = scaler.fit_transform(data[["PJMW_MW"]])
last_24 = data["Scaled"].values[-24:].reshape(1, 24, 1).astype(np.float32)

# Load LSTM model....
@st.cache_resource
def load_lstm_model():
    return tf.keras.models.load_model("model.h5")

model = load_lstm_model()

# Forecast.....
def forecast_lstm(model, input_seq, hours):
    forecast_scaled = []
    seq = input_seq.copy()
    for _ in range(hours):
        pred = model(seq, training=False).numpy()[0][0]
        forecast_scaled.append(pred)
        seq = np.append(seq[0, 1:], [[pred]], axis=0).reshape(1, 24, 1).astype(np.float32)
    return np.array(forecast_scaled).reshape(-1, 1)

forecast_scaled = forecast_lstm(model, last_24, forecast_hours)
forecast_values = scaler.inverse_transform(forecast_scaled)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), periods=forecast_hours, freq='H')
forecast_df = pd.DataFrame({'Forecast_MW': forecast_values.flatten()}, index=forecast_index)

# Plot............
st.subheader(f"🔮 LSTM Forecast for Next {forecast_hours} Hours")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(data["PJMW_MW"].iloc[-168:], label="Last 7 Days Actual")
ax.plot(forecast_df["Forecast_MW"], label="LSTM Forecast")
ax.set_xlabel("Datetime")
ax.set_ylabel("PJMW_MW")
ax.set_title("Energy Consumption Forecast")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Download button..............
st.download_button("📥 Download Forecast CSV", forecast_df.to_csv().encode(),
                   file_name="lstm_forecast.csv", mime="text/csv")
