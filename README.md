# Stock Market Prediction using LSTM and XGBoost

This project implements a hybrid deep learning model that predicts stock prices using Long Short-Term Memory (LSTM) networks combined with XGBoost regression. It is built as an interactive web application using Streamlit, enabling users to visualize stock trends and forecast prices for Apple Inc. (AAPL) based on historical market data.

## 🔍 Overview

The system is designed to:
- Preprocess historical stock data using scaling techniques.
- Use LSTM to capture temporal dependencies in time-series data.
- Apply XGBoost to learn from the residuals of the LSTM model.
- Provide hybrid predictions by combining both model outputs.
- Offer an interactive dashboard for visualization and forecasting.

## 📈 Features

- Real-time stock data fetching using Yahoo Finance.
- Custom target selection (Close, High, Low) for predictions.
- Interactive charts comparing actual, LSTM, and hybrid predictions.
- 30-day forward forecast based on weekday trading.
- Root Mean Squared Error (RMSE) and accuracy evaluation.

## 🚀 Technologies Used

- Python 3.9+
- Streamlit
- TensorFlow (LSTM Model)
- XGBoost (XGBRegressor)
- Pandas & NumPy (Data handling)
- scikit-learn (MinMaxScaler, RMSE)
- yfinance (Stock data API)
- Matplotlib (Data visualization)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/srivanij23/StockPrice-Prediction.git
cd StockPrice-Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## ▶️ Running the App

Run the Streamlit app with:
```bash
streamlit run app.py
```

> Ensure you have a stable internet connection to fetch real-time stock data from Yahoo Finance.

## 📂 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
└── models/                # (Optional) Directory for saved models
```

## 📌 Notes

- Current implementation is focused on AAPL stock, but can be extended to other tickers by modifying the `yfinance` call.
- The hybrid model outperforms standalone LSTM by reducing error and improving forecast stability.

## 🛠️ Future Work

- Multi-stock support and portfolio-level forecasting
- Real-time prediction with streaming data
- Sentiment analysis integration from news or social media
- Advanced hyperparameter tuning and model explainability

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
