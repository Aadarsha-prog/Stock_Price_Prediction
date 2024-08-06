import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt  # Ensure matplotlib is imported

st.title("Stock Price Prediction")

stock = st.text_input("Enter the stock ID", "GOOG")

from datetime import datetime, timedelta
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)
future_end = end + timedelta(days=10)  # 10 days ahead

google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader("Original Close Price and MA for 250 days")
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data))

st.subheader("Original Close Price and MA for 100 days")
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data))

st.subheader("Original Close Price and MA for 200 days and MA for 100 days")
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data, 1, google_data['MA_for_100_days']))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i]) 

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
        'Original_test_data': inv_y_test.reshape(-1),
        'Predictions': inv_predictions.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)
plotting_data.tail()

st.subheader("Original values vs Prediction values")
st.write(plotting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100], plotting_data], axis=0))
plt.legend(["Data -not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Extend predictions for the next 10 days
future_predictions = []
last_100_days = scaled_data[-100:]

for i in range(10):  # Predicting next 10 days
    next_pred = model.predict(last_100_days.reshape(1, 100, 1))
    future_predictions.append(next_pred[0])
    last_100_days = np.append(last_100_days[1:], next_pred, axis=0)

inv_future_predictions = scaler.inverse_transform(future_predictions)

future_dates = pd.date_range(start=end, periods=10)  # Adjusted to 10 days
future_df = pd.DataFrame(inv_future_predictions, index=future_dates, columns=['Future_Prediction'])

st.subheader("Future Predictions for the Next 10 Days")
st.write(future_df)

# Adding residual plot
st.subheader("Residual Plot")
residuals = inv_y_test - inv_predictions
fig = plt.figure(figsize=(15, 6))
plt.scatter(inv_y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
st.pyplot(fig)

# Adding error distribution plot
st.subheader("Error Distribution Plot")
fig = plt.figure(figsize=(15, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
st.pyplot(fig)

# Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(inv_y_test, inv_predictions)
mse = mean_squared_error(inv_y_test, inv_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(inv_y_test, inv_predictions)

st.subheader("Evaluation Metrics")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"R-squared (R2) Score: {r2}")
