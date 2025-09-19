# Stock Analytics Project

A comprehensive Python project for advanced stock analysis and prediction using Yahoo Finance API, machine learning, and interactive web interfaces.

## 🚀 Latest Features

### **🔄 Rolling Backtest (Walk-Forward Validation)**
- **Multi-Period Testing**: Evaluate models across different market conditions
- **No Data Leakage**: Proper time series validation with embargo periods
- **Returns-Based Prediction**: Better stationarity using percentage changes
- **Performance Stability**: Identify consistently performing models
- **Configurable Windows**: Adjustable training, test, and step sizes

### **💰 Investment Calculator**
- **Strategy Comparison**: Buy & Hold vs Model Trading vs Perfect Prediction
- **Real Investment Scenarios**: Calculate potential returns for any amount and period
- **Visual Analysis**: Interactive charts showing strategy performance
- **Risk Assessment**: Comprehensive disclaimers and risk warnings
- **Annualized Returns**: Convert period returns to annual percentages

### **📊 Enhanced Model Performance**
- **Advanced Models**: Gradient Boosting, XGBoost, LightGBM
- **Robust Error Handling**: Improved MAPE calculation with edge case handling
- **Volatility-Based Bounds**: Dynamic prediction constraints based on market volatility
- **Earnings Integration**: Properly scaled earnings features with rarity awareness
- **Feature Scaling**: RobustScaler for better handling of outliers

## Features

- **Data Collection**: Yahoo Finance API integration with earnings data
- **Advanced ML Models**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Technical Indicators**: MA, EMA, MACD, RSI, Bollinger Bands, Volatility measures
- **Earnings Analysis**: Earnings surprises, impact analysis, and calendar integration
- **Interactive Web UI**: Real-time parameter tuning and visualization
- **Investment Analysis**: Practical return calculations and strategy comparison
- **Rolling Backtest**: Walk-forward validation for robust model evaluation

## Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate   # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The system includes optional advanced models (XGBoost, LightGBM) that provide enhanced performance for complex market patterns.

3. Run the analysis:
```bash
# Install the package in development mode
pip install -e .

# Web UI (Recommended)
stockaroo ui             # Launch interactive web dashboard
# or
streamlit run stockaroo/ui/streamlit_app.py

# Command line analysis
stockaroo analyze AAPL  # Analyze Apple stock
stockaroo analyze MSFT --period 2y --models linear_regression lasso

```

## Project Structure

```
stockaroo/
├── stockaroo/                 # Main package
│   ├── __init__.py           # Package initialization
│   ├── cli.py               # Command line interface
│   ├── data/                # Data collection and preprocessing
│   │   ├── collector.py     # Yahoo Finance data collection
│   │   └── preprocessor.py  # Data preprocessing and feature engineering
│   ├── models/              # Machine learning models
│   │   └── predictor.py     # ML models and evaluation
│   ├── ui/                  # User interfaces
│   │   └── streamlit_app.py # Web dashboard
│   ├── utils/               # Utility functions
│   │   └── visualizer.py    # Data visualization
│   └── config/              # Configuration management
│       └── settings.py      # Application settings
├── tests/                   # Test suite
├── docs/                    # Documentation
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
├── main_new.py             # New main entry point
└── README.md               # Project documentation
```

## Current Analysis Results

- **Stock**: Apple (AAPL)
- **Interval**: 1 day
- **Data Period**: 2 years
- **Best Model**: Lasso Regression
- **R² Score**: 0.9432 (94.32% accuracy)
- **RMSE**: $3.19
- **MAPE**: 1.03%

## Model Performance

### **Enhanced Model Suite**
The system now includes 7+ machine learning models with advanced features:

| Model | R² Score | RMSE | MAPE | Features |
|-------|----------|------|------|----------|
| **Lasso** | 0.8984 | $3.71 | 1.17% | Regularized, robust |
| **Ridge** | 0.8473 | $4.55 | 1.43% | Regularized, stable |
| **Linear Regression** | 0.8473 | $4.55 | 1.43% | Baseline model |
| **Random Forest** | 0.8367 | $5.42 | 2.03% | Non-linear, feature importance |
| **Gradient Boosting** | 0.5978 | $7.39 | 2.00% | Advanced ensemble |
| **XGBoost** | Variable | Variable | Variable | High-performance boosting |
| **LightGBM** | Variable | Variable | Variable | Fast, memory-efficient |

### **Performance Improvements:**
- **🔧 Robust Error Handling**: Improved MAPE calculation with edge case handling
- **📊 Volatility-Based Bounds**: Dynamic prediction constraints (2-3 sigma)
- **🎯 Earnings Integration**: Properly scaled earnings features
- **⚡ Feature Scaling**: RobustScaler for better outlier handling
- **🔄 Rolling Validation**: More realistic performance assessment

## Key Features Created

### **📊 Technical Analysis**
- **Moving Averages**: MA(5,10,20), EMA(5,10) with price position indicators
- **Momentum Indicators**: RSI, momentum(5,10,20), price vs MA ratios
- **Volatility Measures**: Rolling volatility(5,10), high-low spreads
- **Volume Analysis**: Volume ratios, volume moving averages
- **Advanced Indicators**: Bollinger Bands, MACD, price momentum

### **🎯 Earnings Integration**
- **Earnings Calendar**: Upcoming earnings announcements
- **Surprise Analysis**: Historical earnings surprises and impacts
- **Price Impact**: Earnings announcement effects on stock prices
- **Proximity Features**: Days since/until earnings with binary indicators
- **Scaled Features**: Rarity-aware scaling for quarterly earnings events

### **🔄 Advanced Validation**
- **Time Series Splits**: Proper chronological data splitting
- **Embargo Periods**: Buffer zones to prevent data leakage
- **Rolling Backtest**: Walk-forward validation across multiple periods
- **Returns-Based Prediction**: Better stationarity using percentage changes
- **Robust Metrics**: Improved MAPE with edge case handling

### **💰 Investment Analysis**
- **Strategy Comparison**: Buy & Hold vs Model Trading vs Perfect Prediction
- **Return Calculations**: Real investment scenario analysis
- **Risk Assessment**: Comprehensive warnings and disclaimers
- **Visual Analysis**: Interactive performance comparison charts

## 🌐 Web Interface Features

The interactive web dashboard provides:

### **📊 Core Analysis**
- **Real-time Stock Data**: Fetch data for any stock symbol with earnings integration
- **Interactive Charts**: Candlestick, line charts with volume and earnings markers
- **Model Comparison**: Compare 7+ ML models side-by-side with performance metrics
- **Prediction Accuracy**: See predictions vs actual values with time series validation

### **🔄 Advanced Features**
- **Rolling Backtest**: Walk-forward validation across multiple time periods
- **Investment Calculator**: Calculate potential returns for different strategies
- **Parameter Tuning**: Real-time adjustment of model parameters and analysis settings
- **Performance Optimization**: AI-powered suggestions for improving model performance

### **⚙️ Interactive Controls**
- **Model Selection**: Choose from Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Advanced Parameters**: Embargo periods, prediction horizons, earnings integration
- **Chart Options**: Multiple chart types, volume display, earnings event markers
- **Investment Settings**: Customizable investment amounts and holding periods

### Launch Web UI
```bash
stockaroo ui
```

## 🚀 Quick Start Guide

### **1. Basic Analysis**
```bash
# Launch the web interface
streamlit run stockaroo/ui/streamlit_app.py

# Select a stock (e.g., AAPL, MSFT, GOOGL)
# Choose models to compare
# Click "Run Analysis"
```

### **2. Enable Rolling Backtest**
```bash
# In the web UI sidebar:
# ✅ Check "Enable Rolling Backtest"
# Set Training Window: 100 days
# Set Test Window: 20 days  
# Set Step Size: 5 days
# Run analysis to see multi-period validation
```

### **3. Investment Calculator**
```bash
# In the web UI sidebar:
# Set Investment Amount: $10,000
# Set Investment Period: 30 days
# Run analysis to see strategy comparison
```

### **4. Advanced Features**
```bash
# Enable earnings integration
# Adjust model parameters (alpha, n_estimators, max_depth)
# Try different chart types (Candlestick, Line, Both)
# Compare with/without earnings data
```

## 🔄 Rolling Backtest System

### **Walk-Forward Validation**
The rolling backtest system provides robust model evaluation by testing across multiple time periods:

```bash
# Enable rolling backtest in the web UI
# Configure: Training Window (50-200 days), Test Window (5-50 days), Step Size (1-20 days)
```

### **Key Benefits:**
- **🎯 Realistic Evaluation**: Tests models across different market conditions
- **🔒 No Data Leakage**: Proper time series validation with embargo periods
- **📈 Performance Stability**: Identifies models that work consistently
- **⚡ Returns-Based**: Uses percentage changes for better stationarity
- **📊 Comprehensive Metrics**: Average performance with standard deviations

### **Results Display:**
- **Performance Summary**: Average metrics across all folds
- **Stability Charts**: R² and MAE stability over time
- **Fold Analysis**: Detailed breakdown of each time period
- **Model Ranking**: Best performing and most stable models

## 💰 Investment Calculator

### **Strategy Comparison**
Calculate potential returns for different investment strategies:

1. **Buy & Hold**: Traditional buy and hold strategy
2. **Model Trading**: Using ML predictions for trading decisions
3. **Perfect Prediction**: Theoretical maximum returns (upper bound)

### **Features:**
- **Customizable Parameters**: Investment amount ($100 - $1M), holding period (1-365 days)
- **Visual Analysis**: Interactive charts comparing strategy performance
- **Risk Assessment**: Comprehensive disclaimers and warnings
- **Annualized Returns**: Convert period returns to annual percentages
- **Performance Insights**: Best/worst strategies with detailed metrics

### **Example Output:**
```
Investment: $10,000 for 30 days
- Buy & Hold: +2.5% return ($250 profit)
- Model Trading: +3.8% return ($380 profit) 
- Perfect Prediction: +8.2% return ($820 profit)
```

## Usage Examples

### Web Interface (Recommended)
```bash
stockaroo ui              # Launch interactive dashboard
```

### Quick Demo
```bash
python demo.py
```

### Full Analysis
```bash
python simple_main.py
```

### Custom Analysis
```python
from stockaroo import StockDataCollector, StockDataPreprocessor, StockPredictor

# Collect data
collector = StockDataCollector("AAPL")
data = collector.get_stock_data(period="1y", interval="1d")

# Preprocess
preprocessor = StockDataPreprocessor()
processed_data = preprocessor.preprocess_pipeline(data)

# Train model
predictor = StockPredictor()
predictor.train_model('linear_regression', processed_data['X_train'], processed_data['y_train'])
results = predictor.evaluate_model('linear_regression', processed_data['X_test'], processed_data['y_test'])

print(f"R² Score: {results['r2']:.4f}")
```

### CLI Commands
```bash
# Analyze a stock
stockaroo analyze AAPL --period 2y --models linear_regression lasso

# Launch basic web interface
stockaroo ui --port 8501

# Launch advanced web interface with cross-market analysis
stockaroo ui --advanced --port 8502

# Show version
stockaroo version
```

## 🔧 **Advanced Prediction System**

Our prediction system uses proper time series analysis with:

### **✅ Key Features:**
- **No data leakage**: Only past information used
- **Realistic constraints**: Max 10% daily price change
- **Proper validation**: Time series splits
- **Realistic performance**: R² scores typically 70-90%

### **🚀 Using the System:**

```bash
# Launch web UI
stockaroo ui --port 8501

# Analyze stocks via CLI
stockaroo analyze AAPL --period 1y --models linear_regression lasso

# Test predictor directly
python -c "from stockaroo.models.predictor import demonstrate_prediction; demonstrate_prediction()"
```

### **📊 System Performance:**

| Feature | Description |
|---------|-------------|
| R² Score | Realistic 70-90% (not overfitted) |
| Next Day Prediction | Bounded by market constraints |
| Data Leakage | None - only past data used |
| Validation | Proper time series splits |
| Constraints | Max 10% daily price change |

## 🌍 **Cross-Market Prediction System**

Advanced multi-market analysis using Hong Kong, European, and American indices:

### **✅ Features:**
- **Multi-Market Data**: Hong Kong (Hang Seng), Europe (STOXX 600), America (S&P 500)
- **Cross-Market Features**: Correlations, lead-lag relationships, volatility analysis
- **Accumulation/Distribution**: Market sentiment and money flow analysis
- **Model Persistence**: Save and load trained models
- **Realistic Predictions**: Bounded by market constraints (max 5% daily change)

### **🚀 Using Cross-Market Prediction:**

```bash
# Test the complete system
python test_cross_market.py

# Use in Python code
from stockaroo.models.cross_market_predictor import CrossMarketPredictor

predictor = CrossMarketPredictor()
X, y, data = predictor.prepare_cross_market_data(period="6mo")
X_train, X_test, y_train, y_test = predictor.time_series_split(X, y)

# Train and evaluate
predictor.train_model('lasso', X_train, y_train)
results = predictor.evaluate_model('lasso', X_test, y_test)

# Predict next day
next_day_pred = predictor.predict_next_day('lasso')
print(f"Predicted US market: {next_day_pred:.2f}")

# Analyze accumulation/distribution
ad_analysis = predictor.analyze_accumulation_distribution()
```

### **📈 Cross-Market Results:**

| Model | R² Score | RMSE | MAPE |
|-------|----------|------|------|
| Linear Regression | 85.5% | 25.17 | 0.30% |
| Ridge | 86.8% | 24.08 | 0.27% |
| Lasso | 88.1% | 22.84 | 0.26% |

### **🔧 Model Persistence:**

```bash
# Models are automatically saved to:
saved_models/                    # Single stock models
saved_models/cross_market/       # Cross-market models

# List saved models
from stockaroo.models.predictor import StockPredictor
predictor = StockPredictor()
saved_models = predictor.list_saved_models()
```

## 🖥️ **Web Interface**

### **Basic UI** (`stockaroo ui`)
- Single stock analysis
- Model performance comparison
- Real-time predictions
- Interactive charts

### **Advanced UI** (`stockaroo ui --advanced`)
- **Single Stock Analysis**: Traditional stock prediction
- **Cross-Market Analysis**: Multi-market prediction system
- **Model Management**: View and manage saved models
- **Accumulation/Distribution**: Market sentiment analysis
- **Cross-Market Correlations**: Market relationship visualization

### **UI Features:**
- 📊 **Interactive Charts**: Plotly-based visualizations
- 🎯 **Real-time Predictions**: Next-day price forecasts
- 📈 **Performance Metrics**: R², RMSE, MAE, MAPE
- 🔧 **Model Management**: Save/load trained models
- 🌍 **Multi-Market View**: HK, EU, US market analysis
- 📱 **Responsive Design**: Works on desktop and mobile

### **Quick Start:**
```bash
# Launch basic UI
python run_advanced_ui.py

# Or use CLI
stockaroo ui --advanced --port 8502
```
