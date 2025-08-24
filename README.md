# Stock Price Prediction using Machine Learning

A comprehensive machine learning pipeline for predicting next-day stock closing prices using historical market data from Yahoo Finance.

## 🚀 Project Overview

This project implements an end-to-end machine learning pipeline that fetches historical stock data, engineers predictive features, and compares multiple models to forecast next-day closing prices. The system focuses on Apple Inc. (AAPL) stock but can be easily adapted for other securities.

## 📊 Key Results

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **Linear Regression** | **$4.19** | **$2.79** | **0.9313** |
| Random Forest | $11.90 | $9.32 | 0.4461 |

Linear Regression significantly outperformed Random Forest, achieving 93.13% explained variance with high accuracy.

## 🔧 Features

- **Data Collection**: Automated fetching of 5 years of historical stock data
- **Feature Engineering**: 10+ predictive features including moving averages, lagged prices, volatility
- **Model Comparison**: Multiple ML algorithms with comprehensive evaluation
- **Visualization**: EDA plots and actual vs. predicted performance charts
- **Automated Pipeline**: Complete workflow from data to results

## 🛠️ Technical Stack

- **Python 3.7+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Data Source**: yfinance (Yahoo Finance)
- **Visualization**: matplotlib
- **Statistical Analysis**: Built-in feature engineering

## 📋 Requirements

```bash
pip install pandas numpy matplotlib yfinance scikit-learn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure
```
stock-price-prediction/
│
├── src/
│   ├── stock_predictor.py      # Main pipeline script
│   └── __pycache__/            # Python cache files
├── results/
│   ├── metrics.txt             # Model performance metrics
│   └── actual_vs_predicted.png # Performance visualization
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```
## 🔍 Feature Engineering

The pipeline creates the following predictive features:

| Feature Category | Features |
|-----------------|----------|
| **Moving Averages** | 5-day, 10-day moving averages |
| **Lagged Prices** | Previous 1, 2, 3 days closing prices |
| **Price Changes** | Absolute and percentage price changes |
| **Volatility** | 10-day rolling standard deviation |
| **Volume Analysis** | Volume moving averages and ratios |

## 📈 Model Performance

### Linear Regression (Best Performance)
- **RMSE**: $4.19 - Low prediction error
- **MAE**: $2.79 - Average error of ~$3 per prediction
- **R²**: 0.9313 - Explains 93.13% of price variance

### Random Forest
- **RMSE**: $11.90 - Higher prediction error
- **MAE**: $9.32 - Less consistent predictions
- **R²**: 0.4461 - Explains 44.61% of price variance

## 📊 Output

```
Stock Price Prediction Results for AAPL
==================================================

Generated on: 2025-08-23 20:00:16

Linear Regression:
  RMSE: $4.19
  MAE: $2.79
  R2: 0.9313

Random Forest:
  RMSE: $11.90
  MAE: $9.32
  R2: 0.4461
```

## 🎯 Business Applications

- **Investment Strategy**: Support for buy/sell decisions
- **Risk Management**: Portfolio risk assessment
- **Market Analysis**: Understanding price patterns
- **Algorithmic Trading**: Component in trading systems

## 🔮 Future Enhancements

- [ ] Multi-stock prediction capability
- [ ] Real-time data integration
- [ ] Advanced deep learning models (LSTM, GRU)
- [ ] Sentiment analysis integration
- [ ] Web dashboard for visualization
- [ ] Automated trading simulation

## Author

**Sharan Karthik**
- GitHub: [sk-0607](https://github.com/sk-0607/)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/sharankarthik06/)
- Email: sharan.675k@gmail.com

