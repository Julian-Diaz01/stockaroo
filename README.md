# Stock Analytics Project

A Python project for stock analysis and prediction using Yahoo Finance API and machine learning.

## Features

- Data collection from Yahoo Finance API
- Stock price analysis and visualization
- Machine learning models for price prediction
- Linear regression, Ridge, Lasso, and Random Forest models
- Technical indicators (MA, MACD, RSI, Bollinger Bands)
- Feature engineering with lag features
- Model performance evaluation and comparison

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

| Model | R² Score | RMSE | MAPE |
|-------|----------|------|------|
| Lasso | 0.9432 | $3.19 | 1.03% |
| Ridge | 0.9417 | $3.24 | 1.05% |
| Linear Regression | 0.9396 | $3.30 | 1.06% |
| Random Forest | 0.8367 | $5.42 | 2.03% |

## Key Features Created

- **Technical Indicators**: Moving averages, MACD, RSI, Bollinger Bands
- **Lag Features**: Previous day prices and returns
- **Price Metrics**: Daily returns, price changes, spreads
- **Volume Analysis**: Volume ratios and moving averages

## 🌐 Web Interface Features

The interactive web dashboard provides:

- **📊 Real-time Stock Data**: Fetch data for any stock symbol
- **⚙️ Parameter Controls**: Adjust analysis parameters in real-time
- **📈 Interactive Charts**: Zoom, pan, and explore price data
- **🤖 Model Comparison**: Compare multiple ML models side-by-side
- **🔍 Feature Analysis**: Visualize feature importance
- **🎯 Prediction Accuracy**: See predictions vs actual values
- **🔮 Future Forecasts**: Get next-day price predictions
- **📊 Technical Indicators**: RSI, MACD, Bollinger Bands, and more

### Launch Web UI
```bash
stockaroo ui
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
