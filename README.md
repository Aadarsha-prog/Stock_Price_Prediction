# Stock_Price_Prediction
This is my College Project assigned by my AI professor.
This project uses a LSTM neural network to predict stock prices based on historical data. The app is built using Streamlit for the frontend.

## Features

- Predict stock prices for the next 10 days.
- Visualize historical stock prices and moving averages.
- Evaluate the prediction performance with various metrics.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Aadarsha-prog/Stock_Price_Prediction.git
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:
```sh
streamlit run web_stock_price_predictor.py
