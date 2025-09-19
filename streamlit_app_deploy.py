"""
Streamlit App for Stock Analytics - Deployment Version
This version uses relative imports and should work better on Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
import importlib
import sys
import os

warnings.filterwarnings('ignore')

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import with different methods
try:
    # Method 1: Direct import
    from stockaroo.data.collector import StockDataCollector
    from stockaroo.models.predictor import StockPredictor
    print("✅ Direct import successful")
except ImportError:
    try:
        # Method 2: Add parent directory and import
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from stockaroo.data.collector import StockDataCollector
        from stockaroo.models.predictor import StockPredictor
        print("✅ Parent directory import successful")
    except ImportError:
        try:
            # Method 3: Install package and import
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
            from stockaroo.data.collector import StockDataCollector
            from stockaroo.models.predictor import StockPredictor
            print("✅ Package install and import successful")
        except Exception as e:
            st.error(f"❌ All import methods failed: {e}")
            st.error("Please check your deployment configuration.")
            st.stop()

# Optional imports for advanced models
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")
        
        # Stock selection
        symbol = st.text_input(
            "Stock Symbol", 
            value="AAPL",
            help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        # Data parameters
        st.subheader("📊 Data Parameters")
        period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        period_value = st.selectbox("Data Period", list(period_options.keys()), index=3)
        period = period_options[period_value]
        
        interval = st.selectbox("Interval", ["1d", "1h", "5m"], index=0)
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = ["Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"]
        
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")
        if LIGHTGBM_AVAILABLE:
            available_models.append("LightGBM")
            
        models_to_use = st.multiselect(
            "Select Models",
            available_models,
            default=["Linear Regression", "Lasso", "Random Forest"]
        )
        
        # Investment Calculator
        st.subheader("💰 Investment Calculator")
        
        investment_amount = st.number_input(
            "Investment Amount ($)", 
            min_value=100, 
            max_value=1000000, 
            value=10000,
            step=1000,
            help="Amount you would invest in the stock"
        )
        
        investment_days = st.number_input(
            "Investment Period (days)", 
            min_value=1, 
            max_value=365, 
            value=30,
            help="Number of days to hold the investment"
        )
        
        # Day Trading Strategy Options
        st.subheader("📈 Day Trading Strategy")
        
        enable_day_trading = st.checkbox(
            "Enable Day Trading Strategy",
            value=False,
            help="Add day trading strategy based on prediction confidence"
        )
        
        if enable_day_trading:
            max_holding_days = st.slider(
                "Max Holding Period (days)",
                min_value=1,
                max_value=30,
                value=5,
                help="Maximum number of days to hold a position"
            )
            
            prediction_threshold = st.slider(
                "Prediction Confidence Threshold (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Minimum predicted price change to trigger a trade"
            )
            
            stop_loss_threshold = st.slider(
                "Stop Loss Threshold (%)",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Maximum loss before selling position"
            )
        else:
            max_holding_days = 1
            prediction_threshold = 1.0
            stop_loss_threshold = 3.0
        
        # Analysis parameters
        st.subheader("🔧 Analysis Parameters")
        lookback_window = st.slider("Lookback Window", 5, 30, 10)
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not symbol:
                st.error("Please enter a stock symbol")
                return
            
            if not models_to_use:
                st.error("Please select at least one model")
                return
            
            # Run analysis
            with st.spinner(f"Running analysis for {symbol}..."):
                try:
                    # Data collection
                    collector = StockDataCollector(symbol)
                    raw_data = collector.get_stock_data(period=period, interval=interval)
                    
                    if raw_data.empty:
                        st.error(f"No data found for {symbol}")
                        return
                    
                    # Model training and evaluation
                    predictor = StockPredictor()
                    X, y = predictor.prepare_time_series_data(raw_data, lookback_window=lookback_window)
                    X_train, X_test, y_train, y_test = predictor.time_series_split(X, y, test_size=test_size/100)
                    
                    results = {}
                    for model_name in models_to_use:
                        model_key = model_name.lower().replace(" ", "_")
                        try:
                            predictor.train_model(model_key, X_train, y_train)
                            eval_results = predictor.evaluate_model(model_key, X_test, y_test)
                            results[model_name] = eval_results
                        except Exception as e:
                            st.warning(f"Failed to train {model_name}: {e}")
                    
                    # Display results
                    if results:
                        st.success("✅ Analysis completed successfully!")
                        
                        # Model performance comparison
                        st.subheader("📊 Model Performance")
                        performance_data = []
                        for model_name, result in results.items():
                            performance_data.append({
                                'Model': model_name,
                                'R² Score': f"{result['r2']:.4f}",
                                'RMSE': f"${result['rmse']:.2f}",
                                'MAE': f"${result['mae']:.2f}",
                                'MAPE': f"{result['mape']:.2f}%"
                            })
                        
                        performance_df = pd.DataFrame(performance_data)
                        st.dataframe(performance_df, use_container_width=True)
                        
                        # Best model
                        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
                        st.success(f"🏆 Best Model: {best_model} (R² = {results[best_model]['r2']:.4f})")
                        
                        # Price chart
                        st.subheader("📈 Price Chart")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=raw_data.index,
                            y=raw_data['Close'],
                            mode='lines',
                            name=f'{symbol} Price',
                            line=dict(color='blue', width=2)
                        ))
                        fig.update_layout(
                            title=f"{symbol} Stock Price",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Predictions vs Actual
                        if best_model in results:
                            st.subheader("🎯 Predictions vs Actual")
                            best_result = results[best_model]
                            
                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(
                                x=list(range(len(y_test))),
                                y=y_test,
                                mode='lines',
                                name='Actual',
                                line=dict(color='blue')
                            ))
                            fig_pred.add_trace(go.Scatter(
                                x=list(range(len(best_result['predictions']))),
                                y=best_result['predictions'],
                                mode='lines',
                                name='Predicted',
                                line=dict(color='red')
                            ))
                            fig_pred.update_layout(
                                title=f"{best_model} - Predictions vs Actual",
                                xaxis_title="Time",
                                yaxis_title="Price ($)",
                                height=400
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.error("Please check your input parameters and try again.")
    
    # Main content area
    st.markdown("""
    ## 🚀 Welcome to Stock Analytics Dashboard
    
    This application provides comprehensive stock analysis using machine learning models.
    
    ### Features:
    - **📊 Real-time Data**: Fetch data from Yahoo Finance
    - **🤖 Multiple Models**: Linear, Ridge, Lasso, Random Forest, and more
    - **📈 Interactive Charts**: Visualize stock prices and predictions
    - **🎯 Performance Metrics**: R², RMSE, MAE, MAPE comparison
    - **🔮 Predictions**: Next-day price forecasts
    
    ### How to Use:
    1. Enter a stock symbol in the sidebar
    2. Select data period and interval
    3. Choose models to compare
    4. Adjust analysis parameters
    5. Click "Run Analysis"
    
    ### Supported Models:
    - **Linear Regression**: Baseline model
    - **Ridge**: Regularized linear model
    - **Lasso**: Feature selection with regularization
    - **Random Forest**: Ensemble method
    - **Gradient Boosting**: Advanced ensemble
    - **XGBoost**: High-performance boosting (if available)
    - **LightGBM**: Fast, memory-efficient boosting (if available)
    """)

if __name__ == "__main__":
    main()
