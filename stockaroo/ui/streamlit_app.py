"""
Improved Streamlit Web UI using the fixed prediction system.
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
warnings.filterwarnings('ignore')

# Force reload modules to ensure latest changes are loaded
if 'stockaroo.data.collector' in sys.modules:
    importlib.reload(sys.modules['stockaroo.data.collector'])
if 'stockaroo.models.predictor' in sys.modules:
    importlib.reload(sys.modules['stockaroo.models.predictor'])

# Import our custom modules
from stockaroo.data.collector import StockDataCollector
from stockaroo.models.predictor import StockPredictor

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
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Show improvement notice
    st.markdown("""
    <div class="success-message">
        ✅ <strong>Advanced Prediction System:</strong> Uses proper time series handling with no data leakage and realistic price constraints.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock symbol input
        symbol = st.text_input(
            "Stock Symbol", 
            value="AAPL", 
            help="Enter a valid stock symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        # Period selection
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        period = st.selectbox("Data Period", list(period_options.keys()))
        period_value = period_options[period]
        
        # Interval selection
        interval = st.selectbox(
            "Data Interval", 
            ["1d", "1wk", "1mo"],
            help="1d = Daily, 1wk = Weekly, 1mo = Monthly"
        )
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = ["Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"]
        
        # Add XGBoost and LightGBM if available
        try:
            import xgboost
            available_models.append("XGBoost")
        except ImportError:
            pass
            
        try:
            import lightgbm
            available_models.append("LightGBM")
        except ImportError:
            pass
        
        models_to_use = st.multiselect(
            "Select Models to Compare",
            available_models,
            default=["Linear Regression", "Lasso", "Gradient Boosting"]
        )
        
        # Lookback window
        lookback_window = st.slider(
            "Lookback Window (days)", 
            min_value=5, 
            max_value=20, 
            value=10,
            help="Number of past days to use as features"
        )
        
        # Test size
        test_size = st.slider(
            "Test Set Size (%)", 
            min_value=10, 
            max_value=40, 
            value=20,
            help="Percentage of data to use for testing"
        ) / 100
        
        # Advanced Parameters
        st.subheader("🔧 Advanced Parameters")
        
        # Embargo period
        embargo_period = st.slider(
            "Embargo Period (days)", 
            min_value=0, 
            max_value=10, 
            value=5,
            help="Buffer days between training and test sets to prevent data leakage"
        )
        
        # Prediction horizon
        prediction_horizon = st.slider(
            "Prediction Horizon (days)", 
            min_value=1, 
            max_value=5, 
            value=1,
            help="How many days ahead to predict"
        )
        
        # Earnings data toggle
        include_earnings = st.checkbox(
            "Include Earnings Data", 
            value=True,
            help="Include earnings features in the model"
        )
        
        # Model-specific parameters
        st.subheader("🎛️ Model Tuning")
        
        # Ridge alpha
        ridge_alpha = st.slider(
            "Ridge Alpha", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0,
            step=0.1,
            help="Regularization strength for Ridge regression"
        )
        
        # Lasso alpha
        lasso_alpha = st.slider(
            "Lasso Alpha", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.1,
            step=0.01,
            help="Regularization strength for Lasso regression"
        )
        
        # Random Forest parameters
        n_estimators = st.slider(
            "Random Forest Trees", 
            min_value=50, 
            max_value=500, 
            value=100,
            step=50,
            help="Number of trees in Random Forest"
        )
        
        max_depth = st.slider(
            "Max Tree Depth", 
            min_value=3, 
            max_value=20, 
            value=10,
            help="Maximum depth of Random Forest trees"
        )
        
        # Chart customization
        st.subheader("📊 Chart Options")
        
        chart_type = st.selectbox(
            "Chart Type",
            ["Candlestick", "Line", "Both"],
            help="Choose the type of price chart to display"
        )
        
        show_volume = st.checkbox(
            "Show Volume", 
            value=True,
            help="Display volume bars on the chart"
        )
        
        show_earnings_markers = st.checkbox(
            "Show Earnings Events", 
            value=True,
            help="Mark earnings announcements on the chart"
        )
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox(
            "Auto-refresh on Parameter Change", 
            value=False,
            help="Automatically re-run analysis when parameters change"
        )
        
        # Rolling Backtest Options
        st.subheader("🔄 Rolling Backtest")
        
        enable_rolling_backtest = st.checkbox(
            "Enable Rolling Backtest", 
            value=False,
            help="Perform walk-forward validation across multiple time periods"
        )
        
        if enable_rolling_backtest:
            backtest_train_size = st.slider(
                "Training Window (days)", 
                min_value=50, 
                max_value=200, 
                value=100,
                help="Number of days to use for training in each fold"
            )
            
            backtest_test_size = st.slider(
                "Test Window (days)", 
                min_value=5, 
                max_value=50, 
                value=20,
                help="Number of days to test in each fold"
            )
            
            backtest_step_size = st.slider(
                "Step Size (days)", 
                min_value=1, 
                max_value=20, 
                value=5,
                help="Number of days to move forward in each iteration"
            )
        else:
            backtest_train_size = 100
            backtest_test_size = 20
            backtest_step_size = 5
        
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
        
        # Run analysis button
        run_analysis_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Clear results button
        if st.session_state.analysis_results:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.analysis_results = None
                st.rerun()
    
    # Main content area
    if run_analysis_btn:
        if not symbol:
            st.error("Please enter a valid stock symbol!")
            return
        
        # Run analysis
        with st.spinner(f"Running improved analysis for {symbol}..."):
            try:
                results = run_analysis(
                    symbol, period_value, interval, 
                    models_to_use, lookback_window, test_size,
                    embargo_period, prediction_horizon, include_earnings,
                    ridge_alpha, lasso_alpha, n_estimators, max_depth,
                    chart_type, show_volume, show_earnings_markers,
                    investment_amount, investment_days,
                    enable_rolling_backtest, backtest_train_size, backtest_test_size, backtest_step_size
                )
                st.session_state.analysis_results = results
                
                # Success message
                st.markdown("""
                <div class="success-message">
                    ✅ Analysis completed successfully! Results show realistic predictions with proper time series handling.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                import traceback
                st.error(f"Error during analysis: {str(e)}")
                st.error(f"Full traceback: {traceback.format_exc()}")
                return
    
    # Display results if available
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)

def run_analysis(symbol, period, interval, models_to_use, lookback_window, test_size, 
                embargo_period=5, prediction_horizon=1, include_earnings=True,
                ridge_alpha=1.0, lasso_alpha=0.1, n_estimators=100, max_depth=10,
                chart_type="Candlestick", show_volume=True, show_earnings_markers=True,
                investment_amount=10000, investment_days=30,
                enable_rolling_backtest=False, backtest_train_size=100, backtest_test_size=20, backtest_step_size=5):
    """Run the improved stock analysis pipeline with earnings integration."""
    
    # Step 1: Data Collection
    collector = StockDataCollector(symbol)
    raw_data = collector.get_stock_data(period=period, interval=interval)
    company_info = collector.get_company_info()
    
    # Step 1.5: Collect earnings data
    try:
        earnings_data = collector.get_earnings_data()
        earnings_impact = collector.get_earnings_impact_analysis(raw_data, earnings_data)
        
        # Combine earnings data
        combined_earnings_data = {
            **earnings_data,
            'impact_analysis': earnings_impact
        }
    except Exception as e:
        st.warning(f"Could not fetch earnings data: {e}")
        earnings_data = {}
        earnings_impact = None
        combined_earnings_data = None
    
    # Step 2: Prepare data using improved method with earnings features
    predictor = StockPredictor()
    
    # Use earnings data only if requested
    earnings_data_to_use = combined_earnings_data if include_earnings else None
    
    X, y = predictor.prepare_time_series_data(
        raw_data, 
        lookback_window=lookback_window, 
        prediction_horizon=prediction_horizon,
        earnings_data=earnings_data_to_use
    )
    
    # Step 2.5: Prepare data without earnings features for comparison (if earnings are included)
    if include_earnings:
        X_no_earnings, y_no_earnings = predictor.prepare_time_series_data(
            raw_data, 
            lookback_window=lookback_window, 
            prediction_horizon=prediction_horizon,
            earnings_data=None
        )
    else:
        X_no_earnings, y_no_earnings = X, y
    
    # Step 3: Split data temporally for both datasets with embargo period
    X_train, X_test, y_train, y_test = predictor.time_series_split(
        X, y, test_size, embargo_period=embargo_period
    )
    X_train_no_earnings, X_test_no_earnings, y_train_no_earnings, y_test_no_earnings = predictor.time_series_split(
        X_no_earnings, y_no_earnings, test_size, embargo_period=embargo_period
    )
    
    # Step 4: Train and evaluate models with earnings data
    model_mapping = {
        "Linear Regression": "linear_regression",
        "Ridge": "ridge", 
        "Lasso": "lasso",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm"
    }
    
    selected_models = [model_mapping[model] for model in models_to_use if model in model_mapping]
    
    eval_results = {}
    eval_results_no_earnings = {}
    
    # Update model parameters
    predictor.models['ridge'].alpha = ridge_alpha
    predictor.models['lasso'].alpha = lasso_alpha
    predictor.models['random_forest'].n_estimators = n_estimators
    predictor.models['random_forest'].max_depth = max_depth
    
    # Update gradient boosting models if available
    if 'gradient_boosting' in predictor.models:
        predictor.models['gradient_boosting'].n_estimators = n_estimators
        predictor.models['gradient_boosting'].max_depth = max_depth
    
    if 'xgboost' in predictor.models:
        predictor.models['xgboost'].n_estimators = n_estimators
        predictor.models['xgboost'].max_depth = max_depth
    
    if 'lightgbm' in predictor.models:
        predictor.models['lightgbm'].n_estimators = n_estimators
        predictor.models['lightgbm'].max_depth = max_depth
    
    for model_name in selected_models:
        try:
            # Train with earnings data
            predictor.train_model(model_name, X_train, y_train)
            eval_results[model_name] = predictor.evaluate_model(
                model_name, X_test, y_test
            )
            
            # Train without earnings data (only if earnings are included)
            if include_earnings:
                predictor.train_model(f"{model_name}_no_earnings", X_train_no_earnings, y_train_no_earnings)
                eval_results_no_earnings[f"{model_name}_no_earnings"] = predictor.evaluate_model(
                    f"{model_name}_no_earnings", X_test_no_earnings, y_test_no_earnings
            )
        except Exception as e:
            st.warning(f"Error with {model_name}: {str(e)}")
    
    # Get best models
    best_model = max(eval_results.keys(), key=lambda x: eval_results[x]['r2']) if eval_results else None
    best_model_no_earnings = max(eval_results_no_earnings.keys(), key=lambda x: eval_results_no_earnings[x]['r2']) if eval_results_no_earnings else None
    
    # Next day predictions with and without earnings data
    next_day_pred = None
    next_day_pred_no_earnings = None
    
    if best_model:
        try:
            recent_data = raw_data.tail(30)  # Use last 30 days
            next_day_pred = predictor.predict_next_day(
                best_model, 
                recent_data, 
                lookback_window,
                earnings_data=earnings_data_to_use
            )
        except Exception as e:
            st.warning(f"Could not make next day prediction with earnings: {e}")
    
    if best_model_no_earnings and include_earnings:
        try:
            recent_data = raw_data.tail(30)  # Use last 30 days
            next_day_pred_no_earnings = predictor.predict_next_day(
                best_model_no_earnings, 
                recent_data, 
                lookback_window,
                earnings_data=None
            )
        except Exception as e:
            st.warning(f"Could not make next day prediction without earnings: {e}")
    
    # Calculate investment returns
    investment_results = calculate_investment_returns(
        raw_data, eval_results, best_model, 
        investment_amount, investment_days, symbol
    )
    
    # Perform rolling backtest if enabled
    rolling_backtest_results = None
    if enable_rolling_backtest:
        try:
            st.info("🔄 Running rolling backtest (walk-forward validation)...")
            rolling_backtest_results = predictor.rolling_backtest(
                raw_data, 
                lookback_window=lookback_window,
                prediction_horizon=prediction_horizon,
                earnings_data=earnings_data_to_use,
                train_size=backtest_train_size,
                test_size=backtest_test_size,
                step_size=backtest_step_size
            )
        except Exception as e:
            st.warning(f"Rolling backtest failed: {e}")
            rolling_backtest_results = None
    
    return {
        'raw_data': raw_data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_test_no_earnings': X_test_no_earnings,
        'y_test_no_earnings': y_test_no_earnings,
        'company_info': company_info,
        'eval_results': eval_results,
        'eval_results_no_earnings': eval_results_no_earnings,
        'best_model': best_model,
        'best_model_no_earnings': best_model_no_earnings,
        'next_day_pred': next_day_pred,
        'next_day_pred_no_earnings': next_day_pred_no_earnings,
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'lookback_window': lookback_window,
        'earnings_data': combined_earnings_data,
        'earnings_impact': earnings_impact,
        'embargo_period': embargo_period,
        'prediction_horizon': prediction_horizon,
        'include_earnings': include_earnings,
        'ridge_alpha': ridge_alpha,
        'lasso_alpha': lasso_alpha,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'chart_type': chart_type,
        'show_volume': show_volume,
        'show_earnings_markers': show_earnings_markers,
        'investment_amount': investment_amount,
        'investment_days': investment_days,
        'investment_results': investment_results,
        'enable_rolling_backtest': enable_rolling_backtest,
        'rolling_backtest_results': rolling_backtest_results
    }

def calculate_investment_returns(raw_data, eval_results, best_model, investment_amount, investment_days, symbol):
    """Calculate potential investment returns based on model predictions."""
    
    if not eval_results or not best_model or best_model not in eval_results:
        return None
    
    try:
        # Get the best model's predictions
        best_model_results = eval_results[best_model]
        y_test = best_model_results.get('predictions', [])
        
        if len(y_test) == 0:
            return None
        
        # Get actual prices for comparison
        test_start_idx = len(raw_data) - len(y_test)
        actual_prices = raw_data.iloc[test_start_idx:]['Close'].values
        
        if len(actual_prices) != len(y_test):
            return None
        
        # Calculate returns for different scenarios
        results = {}
        
        # 1. Buy and Hold (using actual prices)
        if len(actual_prices) >= investment_days:
            buy_price = actual_prices[0]
            sell_price = actual_prices[min(investment_days - 1, len(actual_prices) - 1)]
            shares_bought = investment_amount / buy_price
            buy_hold_value = shares_bought * sell_price
            buy_hold_return = buy_hold_value - investment_amount
            buy_hold_return_pct = (buy_hold_return / investment_amount) * 100
            
            results['buy_hold'] = {
                'strategy': 'Buy & Hold (Actual)',
                'initial_value': investment_amount,
                'final_value': buy_hold_value,
                'return_amount': buy_hold_return,
                'return_percentage': buy_hold_return_pct,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'shares': shares_bought
            }
        
        # 2. Model-based trading (using predictions)
        if len(y_test) >= investment_days:
            # Simple strategy: buy if next day prediction is higher than current price
            trading_returns = []
            current_shares = 0
            current_cash = investment_amount
            
            for i in range(min(investment_days, len(y_test) - 1)):
                current_price = actual_prices[i]
                predicted_price = y_test[i + 1] if i + 1 < len(y_test) else y_test[i]
                
                # Simple trading logic: buy if prediction is higher, sell if lower
                if predicted_price > current_price and current_cash > 0:
                    # Buy with all available cash
                    shares_to_buy = current_cash / current_price
                    current_shares += shares_to_buy
                    current_cash = 0
                elif predicted_price < current_price and current_shares > 0:
                    # Sell all shares
                    current_cash = current_shares * current_price
                    current_shares = 0
            
            # Final value calculation
            if current_shares > 0:
                final_price = actual_prices[min(investment_days - 1, len(actual_prices) - 1)]
                final_value = current_shares * final_price
            else:
                final_value = current_cash
            
            model_return = final_value - investment_amount
            model_return_pct = (model_return / investment_amount) * 100
            
            results['model_trading'] = {
                'strategy': f'Model Trading ({best_model.replace("_", " ").title()})',
                'initial_value': investment_amount,
                'final_value': final_value,
                'return_amount': model_return,
                'return_percentage': model_return_pct,
                'shares_held': current_shares,
                'cash_remaining': current_cash
            }
        
        # 3. Perfect prediction (upper bound)
        if len(actual_prices) >= investment_days:
            # Find the best possible buy and sell points
            best_buy_idx = 0
            best_sell_idx = min(investment_days - 1, len(actual_prices) - 1)
            
            for i in range(min(investment_days, len(actual_prices))):
                for j in range(i + 1, min(investment_days, len(actual_prices))):
                    if actual_prices[j] / actual_prices[i] > actual_prices[best_sell_idx] / actual_prices[best_buy_idx]:
                        best_buy_idx = i
                        best_sell_idx = j
            
            perfect_buy_price = actual_prices[best_buy_idx]
            perfect_sell_price = actual_prices[best_sell_idx]
            perfect_shares = investment_amount / perfect_buy_price
            perfect_value = perfect_shares * perfect_sell_price
            perfect_return = perfect_value - investment_amount
            perfect_return_pct = (perfect_return / investment_amount) * 100
            
            results['perfect_prediction'] = {
                'strategy': 'Perfect Prediction (Upper Bound)',
                'initial_value': investment_amount,
                'final_value': perfect_value,
                'return_amount': perfect_return,
                'return_percentage': perfect_return_pct,
                'buy_price': perfect_buy_price,
                'sell_price': perfect_sell_price,
                'shares': perfect_shares
            }
        
        return results
        
    except Exception as e:
        st.warning(f"Error calculating investment returns: {e}")
        return None

def display_results(results):
    """Display improved analysis results."""
    
    # Company information
    st.header("📊 Company Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Company", results['company_info'].get('name', 'N/A'))
    with col2:
        st.metric("Sector", results['company_info'].get('sector', 'N/A'))
    with col3:
        st.metric("Industry", results['company_info'].get('industry', 'N/A'))
    with col4:
        st.metric("Data Points", len(results['raw_data']))
    
    # Analysis Configuration Summary
    st.header("⚙️ Analysis Configuration")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lookback Window", f"{results['lookback_window']} days")
        st.metric("Embargo Period", f"{results['embargo_period']} days")
    with col2:
        st.metric("Test Size", f"{results.get('test_size', 0.2)*100:.0f}%")
        st.metric("Prediction Horizon", f"{results['prediction_horizon']} day(s)")
    with col3:
        st.metric("Ridge Alpha", f"{results['ridge_alpha']:.2f}")
        st.metric("Lasso Alpha", f"{results['lasso_alpha']:.3f}")
    with col4:
        st.metric("RF Trees", results['n_estimators'])
        st.metric("Max Depth", results['max_depth'])
    
    # Earnings and Chart Configuration
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Earnings Data", "✅ Included" if results['include_earnings'] else "❌ Excluded")
    with col2:
        st.metric("Chart Type", results['chart_type'])
    
    # Price chart with earnings markers
    st.header("📈 Stock Price Chart with Earnings Events")
    earnings_impact = results.get('earnings_impact', None)
    
    # Get chart parameters from results (with defaults)
    chart_type = results.get('chart_type', 'Candlestick')
    show_volume = results.get('show_volume', True)
    show_earnings_markers = results.get('show_earnings_markers', True)
    
    if show_earnings_markers and earnings_impact is not None:
        create_price_chart_with_earnings(results['raw_data'], results['symbol'], earnings_impact)
    else:
        create_price_chart(results['raw_data'], results['symbol'], chart_type, show_volume)
    
    # Earnings analysis
    if earnings_impact is not None and not earnings_impact.empty:
        st.header("💰 Earnings Impact Analysis")
        display_earnings_analysis(earnings_impact)
    else:
        st.header("💰 Earnings Impact Analysis")
        st.info("No earnings data available for this symbol. Earnings features will use default values in the model.")
    
    # Model performance
    st.header("🤖 Model Performance")
    display_model_performance(results['eval_results'], results['best_model'])
    
    # Interactive Performance Optimization
    st.header("🎛️ Performance Optimization Insights")
    display_performance_optimization(results)
    
    # Rolling Backtest Results
    if results.get('enable_rolling_backtest') and results.get('rolling_backtest_results'):
        st.header("🔄 Rolling Backtest Results")
        display_rolling_backtest_results(results)
    
    # Investment Calculator Results
    if results.get('investment_results'):
        st.header("💰 Investment Calculator Results")
        display_investment_results(results)
    
    # Predictions vs Actual
    st.header("🎯 Predictions vs Actual (Time Series Split)")
    display_predictions_comparison(results)
    
    # Next day predictions comparison
    if results['next_day_pred'] is not None or results['next_day_pred_no_earnings'] is not None:
        st.header("🔮 Next Day Predictions Comparison")
        display_next_day_predictions_comparison(results)
    
    # Earnings and Prediction Comparison
    st.header("📊 Earnings Events & Price Predictions")
    display_earnings_and_predictions(results)
    
    # Analysis explanation
    st.header("📚 Analysis Explanation")
    display_analysis_explanation()

def create_price_chart(data, symbol, chart_type="Candlestick", show_volume=True):
    """Create interactive price chart with customizable options."""
    
    fig = go.Figure()
    
    # Ensure data index is properly formatted for Plotly
    data_copy = data.copy()
    if not isinstance(data_copy.index, pd.DatetimeIndex):
        data_copy.index = pd.to_datetime(data_copy.index)
    
    # Add price chart based on type
    if chart_type in ["Candlestick", "Both"]:
        try:
            fig.add_trace(go.Candlestick(
                x=data_copy.index,
                open=data_copy['Open'],
                high=data_copy['High'],
                low=data_copy['Low'],
                close=data_copy['Close'],
                name=f"{symbol} Price"
            ))
        except Exception as e:
            st.warning(f"Could not create candlestick chart: {e}. Using line chart instead.")
            chart_type = "Line"
    
    if chart_type in ["Line", "Both"]:
        fig.add_trace(go.Scatter(
            x=data_copy.index,
            y=data_copy['Close'],
            mode='lines',
            name=f"{symbol} Close",
            line=dict(color='blue', width=2)
        ))
    
    # Add volume if requested
    if show_volume and 'Volume' in data_copy.columns:
        # Create secondary y-axis for volume
        fig.add_trace(go.Bar(
            x=data_copy.index,
            y=data_copy['Volume'],
            name="Volume",
            yaxis="y2",
            opacity=0.3,
            marker_color='lightblue'
        ))
        
        # Update layout for dual y-axis
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right"
            )
        )
    
    fig.update_layout(
        title=f"{symbol} Stock Price Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_price_chart_with_earnings(data, symbol, earnings_impact):
    """Create interactive price chart with earnings event markers and tooltips."""
    
    fig = go.Figure()
    
    # Ensure data index is properly formatted for Plotly
    data_copy = data.copy()
    if not isinstance(data_copy.index, pd.DatetimeIndex):
        data_copy.index = pd.to_datetime(data_copy.index)
    
    # Candlestick chart with error handling
    try:
        fig.add_trace(go.Candlestick(
            x=data_copy.index,
            open=data_copy['Open'],
            high=data_copy['High'],
            low=data_copy['Low'],
            close=data_copy['Close'],
        name=symbol
        ))
    except Exception as e:
        # Fallback to line chart if candlestick fails
        st.warning(f"Could not create candlestick chart: {e}. Using line chart instead.")
        fig.add_trace(go.Scatter(
            x=data_copy.index,
            y=data_copy['Close'],
            mode='lines',
            name=symbol,
            line=dict(color='blue', width=2)
        ))
    
    # Add earnings event markers if available
    if earnings_impact is not None and not earnings_impact.empty:
        for idx, row in earnings_impact.iterrows():
            earnings_date = row['earnings_date']
            surprise_pct = row.get('earnings_surprise_pct', 0) or 0
            price_change_pct = row.get('price_change_pct', 0) or 0
            actual_eps = row.get('actual_eps', 'N/A')
            estimated_eps = row.get('estimated_eps', 'N/A')
            
            # Determine marker color based on surprise
            if surprise_pct > 0:
                marker_color = 'green'
                surprise_text = f"Beat by {surprise_pct:.1f}%"
            elif surprise_pct < 0:
                marker_color = 'red'
                surprise_text = f"Missed by {abs(surprise_pct):.1f}%"
            else:
                marker_color = 'orange'
                surprise_text = "Met expectations"
            
            # Find the closest price data point
            closest_price = None
            for price_date in data_copy.index:
                if abs((price_date - earnings_date).days) <= 1:
                    closest_price = data_copy.loc[price_date, 'Close']
                    break
            
            if closest_price is not None:
                fig.add_trace(go.Scatter(
                    x=[earnings_date],
                    y=[closest_price],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=marker_color,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    name=f"Earnings: {surprise_text}",
                    hovertemplate="<b>Earnings Event</b><br>" +
                                 f"Date: {earnings_date.strftime('%Y-%m-%d')}<br>" +
                                 f"Actual EPS: {actual_eps}<br>" +
                                 f"Estimated EPS: {estimated_eps}<br>" +
                                 f"Surprise: {surprise_pct:.1f}%<br>" +
                                 f"Price Impact: {price_change_pct:.1f}%<br>" +
                                 f"Price: ${closest_price:.2f}<br>" +
                                 "<extra></extra>",
                    showlegend=False
    ))
    
    fig.update_layout(
        title=f"{symbol} Stock Price with Earnings Events",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_earnings_analysis(earnings_impact):
    """Display earnings impact analysis."""
    
    if earnings_impact.empty:
        st.warning("No earnings data available for analysis.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Earnings Events", len(earnings_impact))
    
    with col2:
        avg_surprise = earnings_impact['earnings_surprise_pct'].mean()
        st.metric("Avg Earnings Surprise", f"{avg_surprise:.1f}%")
    
    with col3:
        avg_impact = earnings_impact['price_change_pct'].mean()
        st.metric("Avg Price Impact", f"{avg_impact:.1f}%")
    
    with col4:
        beats = len(earnings_impact[earnings_impact['earnings_surprise_pct'] > 0])
        beat_rate = (beats / len(earnings_impact)) * 100
        st.metric("Beat Rate", f"{beat_rate:.1f}%")
    
    # Earnings events table
    st.subheader("📊 Recent Earnings Events")
    
    # Prepare display data
    display_data = earnings_impact.copy()
    display_data['earnings_date'] = display_data['earnings_date'].dt.strftime('%Y-%m-%d')
    display_data = display_data.rename(columns={
        'earnings_date': 'Date',
        'actual_eps': 'Actual EPS',
        'estimated_eps': 'Estimated EPS',
        'earnings_surprise_pct': 'Surprise %',
        'price_change_pct': 'Price Impact %',
        'pre_earnings_price': 'Pre-Earnings Price',
        'post_earnings_price': 'Post-Earnings Price'
    })
    
    # Select columns to display
    display_columns = ['Date', 'Actual EPS', 'Estimated EPS', 'Surprise %', 'Price Impact %']
    display_df = display_data[display_columns].tail(10)  # Show last 10 events
    
    # Format the dataframe
    styled_df = display_df.style.format({
        'Actual EPS': '{:.2f}',
        'Estimated EPS': '{:.2f}',
        'Surprise %': '{:.1f}%',
        'Price Impact %': '{:.1f}%'
    })
    
    # Color code surprise column
    def highlight_surprise(val):
        if val > 0:
            return 'background-color: #d4edda'  # Light green
        elif val < 0:
            return 'background-color: #f8d7da'  # Light red
        else:
            return 'background-color: #fff3cd'  # Light yellow
    
    styled_df = styled_df.applymap(highlight_surprise, subset=['Surprise %'])
    
    st.dataframe(styled_df, width='stretch')
    
    # Earnings surprise vs price impact scatter plot
    st.subheader("📈 Earnings Surprise vs Price Impact")
    
    fig = px.scatter(
        earnings_impact,
        x='earnings_surprise_pct',
        y='price_change_pct',
        title='Earnings Surprise vs Stock Price Impact',
        labels={
            'earnings_surprise_pct': 'Earnings Surprise (%)',
            'price_change_pct': 'Price Change (%)'
        },
        hover_data=['earnings_date', 'actual_eps', 'estimated_eps'],
        color='earnings_surprise_pct',
        color_continuous_scale='RdYlGn'
    )
    
    # Add correlation line
    correlation = earnings_impact['earnings_surprise_pct'].corr(earnings_impact['price_change_pct'])
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f'Correlation: {correlation:.3f}',
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def display_model_performance(eval_results, best_model):
    """Display improved model performance metrics."""
    
    if not eval_results:
        st.warning("No model results available.")
        return
    
    # Create performance dataframe
    performance_data = []
    for model_name, results in eval_results.items():
        performance_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'R² Score': results['r2'],
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'MAPE (%)': results['mape']
        })
    
    df = pd.DataFrame(performance_data)
    
    # Highlight best model
    def highlight_best(row):
        if row['Model'].lower().replace(' ', '_') == best_model:
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df.style.apply(highlight_best, axis=1).format({
            'R² Score': '{:.4f}',
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}',
            'MAPE (%)': '{:.2f}'
        }),
        width='stretch'
    )
    
    # Performance chart
    fig = px.bar(
        df, 
        x='Model', 
        y='R² Score',
        title='Model Performance Comparison (Realistic R² Scores)',
        color='R² Score',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance explanation
    st.markdown("""
    <div class="warning-message">
        <strong>Note:</strong> These R² scores are realistic (typically 70-90%) because we use proper time series validation with no data leakage.
    </div>
    """, unsafe_allow_html=True)

def display_predictions_comparison(results):
    """Display improved predictions vs actual values."""
    
    if not results['eval_results'] or not results['best_model']:
        st.warning("No prediction results available.")
        return
    
    best_model_results = results['eval_results'][results['best_model']]
    y_test = results['y_test']
    y_pred = best_model_results['predictions']
    
    # Create comparison dataframe
    comparison_data = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': np.abs(y_test - y_pred),
        'Error %': (np.abs(y_test - y_pred) / y_test) * 100
    })
    
    # Display last 20 predictions
    st.subheader(f"Last 20 Predictions ({results['best_model'].replace('_', ' ').title()})")
    st.dataframe(
        comparison_data.tail(20).style.format({
            'Actual': '{:.2f}',
            'Predicted': '{:.2f}',
            'Error': '{:.2f}',
            'Error %': '{:.1f}%'
        }),
        width='stretch'
    )
    
    # Scatter plot
    fig = px.scatter(
        comparison_data,
        x='Actual',
        y='Predicted',
        title='Predictions vs Actual Values (Time Series Split)',
        color='Error %',
        color_continuous_scale='reds'
    )
    
    # Add perfect prediction line
    min_val = min(comparison_data['Actual'].min(), comparison_data['Predicted'].min())
    max_val = max(comparison_data['Actual'].max(), comparison_data['Predicted'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='black')
    ))
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def display_next_day_prediction(results):
    """Display realistic next day prediction."""
    
    current_price = results['raw_data'].iloc[-1]['Close']
    next_day_pred = results['next_day_pred']
    change_pct = ((next_day_pred - current_price) / current_price) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Predicted Tomorrow", f"${next_day_pred:.2f}")
    with col3:
        st.metric("Expected Change", f"{change_pct:+.2f}%")
    
    # Prediction explanation
    st.markdown("""
    <div class="success-message">
        <strong>Realistic Prediction:</strong> This prediction is bounded by realistic market constraints (max 10% daily change) and uses only past data.
    </div>
    """, unsafe_allow_html=True)

def display_next_day_predictions_comparison(results):
    """Display comparison of next day predictions with and without earnings data."""
    
    current_price = results['raw_data'].iloc[-1]['Close']
    next_day_pred = results.get('next_day_pred', None)
    next_day_pred_no_earnings = results.get('next_day_pred_no_earnings', None)
    
    if next_day_pred is None and next_day_pred_no_earnings is None:
        st.warning("No next day predictions available.")
        return
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        if next_day_pred is not None:
            change_pct = ((next_day_pred - current_price) / current_price) * 100
            st.metric("Predicted (With Earnings)", f"${next_day_pred:.2f}", f"{change_pct:+.2f}%")
        else:
            st.metric("Predicted (With Earnings)", "N/A")
    
    with col3:
        if next_day_pred_no_earnings is not None:
            change_pct = ((next_day_pred_no_earnings - current_price) / current_price) * 100
            st.metric("Predicted (Without Earnings)", f"${next_day_pred_no_earnings:.2f}", f"{change_pct:+.2f}%")
        else:
            st.metric("Predicted (Without Earnings)", "N/A")
    
    # Show difference between predictions
    if next_day_pred is not None and next_day_pred_no_earnings is not None:
        difference = next_day_pred - next_day_pred_no_earnings
        difference_pct = (difference / current_price) * 100
        
        st.subheader("📊 Prediction Difference Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Absolute Difference", f"${abs(difference):.2f}")
        
        with col2:
            st.metric("Percentage Difference", f"{abs(difference_pct):.2f}%")
        
        # Interpretation
        if abs(difference_pct) > 1:
            st.info(f"💡 The earnings data makes a significant difference in the prediction ({difference_pct:+.2f}%).")
        else:
            st.info(f"💡 The earnings data has a minimal impact on the prediction ({difference_pct:+.2f}%).")
    
    # Prediction explanation
    st.markdown("""
    <div class="success-message">
        <strong>Dual Prediction System:</strong> 
        - <strong>With Earnings:</strong> Uses earnings data, surprise history, and impact analysis
        - <strong>Without Earnings:</strong> Uses only price and technical indicators
        - Both predictions are bounded by realistic market constraints (max 10% daily change)
    </div>
    """, unsafe_allow_html=True)

def display_earnings_and_predictions(results):
    """Display earnings events and price predictions comparison."""
    
    # Get data
    raw_data = results['raw_data']
    earnings_impact = results.get('earnings_impact', None)
    eval_results = results['eval_results']
    eval_results_no_earnings = results.get('eval_results_no_earnings', {})
    best_model = results['best_model']
    best_model_no_earnings = results.get('best_model_no_earnings', None)
    
    if not eval_results or not best_model:
        st.warning("No prediction results available.")
        return
    
    # Get predictions
    best_model_results = eval_results[best_model]
    y_test = results['y_test']
    y_pred = best_model_results['predictions']
    
    # Get predictions without earnings
    y_pred_no_earnings = None
    if best_model_no_earnings and best_model_no_earnings in eval_results_no_earnings:
        best_model_no_earnings_results = eval_results_no_earnings[best_model_no_earnings]
        y_pred_no_earnings = best_model_no_earnings_results['predictions']
    
    # Create a simple chart for actual vs predicted prices
    fig = go.Figure()
    
    # Prepare data
    data_copy = raw_data.copy()
    if not isinstance(data_copy.index, pd.DatetimeIndex):
        data_copy.index = pd.to_datetime(data_copy.index)
    
    # Get the test period dates
    test_start_idx = len(raw_data) - len(y_test)
    test_dates = data_copy.index[test_start_idx:]
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    # Add predicted prices with earnings
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred,
        mode='lines+markers',
        name='Predicted (With Earnings)',
        line=dict(color='green', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add predicted prices without earnings
    if y_pred_no_earnings is not None:
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_pred_no_earnings,
            mode='lines+markers',
            name='Predicted (Without Earnings)',
            line=dict(color='red', width=3, dash='dot'),
            marker=dict(size=6)
        ))
    
    # Add earnings markers if available
    if earnings_impact is not None and not earnings_impact.empty:
        for idx, row in earnings_impact.iterrows():
            earnings_date = row['earnings_date']
            surprise_pct = row.get('earnings_surprise_pct', 0) or 0
            actual_eps = row.get('actual_eps', 'N/A')
            estimated_eps = row.get('estimated_eps', 'N/A')
            
            # Only show earnings events that are in the test period
            if earnings_date in test_dates:
                # Determine marker color based on surprise
                if surprise_pct > 0:
                    marker_color = 'green'
                    surprise_text = f"Beat by {surprise_pct:.1f}%"
                elif surprise_pct < 0:
                    marker_color = 'red'
                    surprise_text = f"Missed by {abs(surprise_pct):.1f}%"
                else:
                    marker_color = 'orange'
                    surprise_text = "Met expectations"
                
                # Find the closest price data point
                closest_price = None
                for price_date in test_dates:
                    if abs((price_date - earnings_date).days) <= 1:
                        closest_price = y_test[test_dates == price_date].iloc[0] if len(y_test[test_dates == price_date]) > 0 else None
                        break
                
                if closest_price is not None:
                    fig.add_trace(go.Scatter(
                        x=[earnings_date],
                        y=[closest_price],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=marker_color,
                            symbol='diamond',
                            line=dict(width=3, color='white')
                        ),
                        name=f"Earnings: {surprise_text}",
                        hovertemplate="<b>Earnings Event</b><br>" +
                                     f"Date: {earnings_date.strftime('%Y-%m-%d')}<br>" +
                                     f"Actual EPS: {actual_eps}<br>" +
                                     f"Estimated EPS: {estimated_eps}<br>" +
                                     f"Surprise: {surprise_pct:.1f}%<br>" +
                                     f"Price: ${closest_price:.2f}<br>" +
                                     "<extra></extra>",
                        showlegend=False
                    ))
    
    # Update layout
    fig.update_layout(
        height=600,
        title=f"{results['symbol']} - Predictions Comparison: With vs Without Earnings Data",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display earnings summary table
    if earnings_impact is not None and not earnings_impact.empty:
        st.subheader("📋 Earnings Events Summary")
        
        # Create summary table
        summary_data = []
        for idx, row in earnings_impact.iterrows():
            earnings_date = row['earnings_date']
            surprise_pct = row.get('earnings_surprise_pct', 0) or 0
            price_impact = row.get('price_change_pct', 0) or 0
            actual_eps = row.get('actual_eps', 'N/A')
            estimated_eps = row.get('estimated_eps', 'N/A')
            
            # Determine if this earnings event was in the test period
            test_period = earnings_date in test_dates if len(test_dates) > 0 else False
            
            summary_data.append({
                'Date': earnings_date.strftime('%Y-%m-%d'),
                'Actual EPS': f"{actual_eps:.2f}" if isinstance(actual_eps, (int, float)) else str(actual_eps),
                'Estimated EPS': f"{estimated_eps:.2f}" if isinstance(estimated_eps, (int, float)) else str(estimated_eps),
                'Surprise %': f"{surprise_pct:.1f}%",
                'Price Impact %': f"{price_impact:.1f}%",
                'In Test Period': 'Yes' if test_period else 'No'
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Color code the surprise column
            def highlight_surprise(val):
                val_num = float(val.replace('%', ''))
                if val_num > 0:
                    return 'background-color: #d4edda'  # Light green
                elif val_num < 0:
                    return 'background-color: #f8d7da'  # Light red
                else:
                    return 'background-color: #fff3cd'  # Light yellow
            
            styled_summary = summary_df.style.applymap(highlight_surprise, subset=['Surprise %'])
            st.dataframe(styled_summary, width='stretch')
    
    # Display prediction accuracy metrics comparison
    st.subheader("🎯 Prediction Accuracy Metrics Comparison")
    
    # Calculate metrics for both models
    mae_with = np.mean(np.abs(y_test - y_pred))
    rmse_with = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape_with = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    actual_direction = np.diff(y_test) > 0
    pred_direction_with = np.diff(y_pred) > 0
    directional_accuracy_with = np.mean(actual_direction == pred_direction_with) * 100
    
    if y_pred_no_earnings is not None:
        mae_without = np.mean(np.abs(y_test - y_pred_no_earnings))
        rmse_without = np.sqrt(np.mean((y_test - y_pred_no_earnings) ** 2))
        mape_without = np.mean(np.abs((y_test - y_pred_no_earnings) / y_test)) * 100
        pred_direction_without = np.diff(y_pred_no_earnings) > 0
        directional_accuracy_without = np.mean(actual_direction == pred_direction_without) * 100
    
    # Create comparison table
    metrics_data = {
        'Metric': ['Mean Absolute Error', 'Root Mean Square Error', 'Mean Absolute Percentage Error', 'Directional Accuracy'],
        'With Earnings': [
            f"${mae_with:.2f}",
            f"${rmse_with:.2f}",
            f"{mape_with:.1f}%",
            f"{directional_accuracy_with:.1f}%"
        ]
    }
    
    if y_pred_no_earnings is not None:
        metrics_data['Without Earnings'] = [
            f"${mae_without:.2f}",
            f"${rmse_without:.2f}",
            f"{mape_without:.1f}%",
            f"{directional_accuracy_without:.1f}%"
        ]
        
        # Calculate improvement
        mae_improvement = ((mae_without - mae_with) / mae_without) * 100
        rmse_improvement = ((rmse_without - rmse_with) / rmse_without) * 100
        mape_improvement = ((mape_without - mape_with) / mape_without) * 100
        directional_improvement = directional_accuracy_with - directional_accuracy_without
        
        metrics_data['Improvement'] = [
            f"{mae_improvement:+.1f}%",
            f"{rmse_improvement:+.1f}%",
            f"{mape_improvement:+.1f}%",
            f"{directional_improvement:+.1f}%"
        ]
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, width='stretch')
    
    # Show which model is better
    if y_pred_no_earnings is not None:
        if mae_with < mae_without:
            st.success("✅ Model with earnings data performs better on most metrics!")
        else:
            st.info("ℹ️ Model without earnings data performs better on most metrics.")

def display_performance_optimization(results):
    """Display performance optimization insights and suggestions."""
    
    eval_results = results['eval_results']
    best_model = results['best_model']
    
    if not eval_results or not best_model:
        st.warning("No performance data available for optimization insights.")
        return
    
    # Get best model performance
    best_performance = eval_results[best_model]
    r2 = best_performance['r2']
    mae = best_performance['mae']
    rmse = best_performance['rmse']
    mape = best_performance['mape']
    
    # Performance analysis
    st.subheader("📊 Current Performance Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if r2 > 0.8:
            st.success(f"R²: {r2:.3f} (Excellent)")
        elif r2 > 0.6:
            st.info(f"R²: {r2:.3f} (Good)")
        else:
            st.warning(f"R²: {r2:.3f} (Needs Improvement)")
    
    with col2:
        if mape < 5:
            st.success(f"MAPE: {mape:.1f}% (Excellent)")
        elif mape < 10:
            st.info(f"MAPE: {mape:.1f}% (Good)")
        else:
            st.warning(f"MAPE: {mape:.1f}% (Needs Improvement)")
    
    with col3:
        if mae < 1:
            st.success(f"MAE: ${mae:.2f} (Excellent)")
        elif mae < 3:
            st.info(f"MAE: ${mae:.2f} (Good)")
        else:
            st.warning(f"MAE: ${mae:.2f} (Needs Improvement)")
    
    with col4:
        if rmse < 2:
            st.success(f"RMSE: ${rmse:.2f} (Excellent)")
        elif rmse < 5:
            st.info(f"RMSE: ${rmse:.2f} (Good)")
        else:
            st.warning(f"RMSE: ${rmse:.2f} (Needs Improvement)")
    
    # Optimization suggestions
    st.subheader("💡 Optimization Suggestions")
    
    suggestions = []
    
    # R² suggestions
    if r2 < 0.6:
        suggestions.append("🔧 **Low R²**: Try increasing lookback window or adding more features")
    
    # MAPE suggestions
    if mape > 10:
        suggestions.append("📈 **High MAPE**: Consider using ensemble methods or feature engineering")
    
    # Model-specific suggestions
    if best_model == 'ridge' and results['ridge_alpha'] < 0.5:
        suggestions.append("⚖️ **Ridge Model**: Try increasing alpha for better regularization")
    elif best_model == 'lasso' and results['lasso_alpha'] < 0.05:
        suggestions.append("🎯 **Lasso Model**: Try increasing alpha for better feature selection")
    elif best_model in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
        if results['n_estimators'] < 200:
            suggestions.append(f"🌲 **{best_model.replace('_', ' ').title()}**: Try increasing number of trees")
        if results['max_depth'] > 15:
            suggestions.append(f"🌲 **{best_model.replace('_', ' ').title()}**: Try reducing max depth to prevent overfitting")
    
    # Gradient boosting specific suggestions
    if best_model in ['gradient_boosting', 'xgboost', 'lightgbm']:
        suggestions.append("🚀 **Gradient Boosting**: These models handle non-linearity well and should reduce systematic bias")
    
    # Data suggestions
    if not results['include_earnings']:
        suggestions.append("💰 **Earnings Data**: Enable earnings features for potentially better predictions")
    
    if results['embargo_period'] < 3:
        suggestions.append("⏰ **Data Leakage**: Consider increasing embargo period")
    
    if results['lookback_window'] < 15:
        suggestions.append("📅 **Lookback Window**: Try increasing for more historical context")
    
    # Display suggestions
    if suggestions:
        for suggestion in suggestions:
            st.markdown(suggestion)
    else:
        st.success("🎉 Your model is performing well! Current parameters seem optimal.")
    
    # Parameter comparison table
    st.subheader("📋 Parameter Impact Analysis")
    
    # Create a comparison of all models
    model_comparison = []
    for model_name, performance in eval_results.items():
        # Calculate directional accuracy if not present
        directional_accuracy = performance.get('directional_accuracy', None)
        if directional_accuracy is None:
            # Calculate directional accuracy from predictions
            y_test = results['y_test']
            y_pred = performance['predictions']
            if len(y_test) > 1 and len(y_pred) > 1:
                # Calculate direction accuracy
                actual_direction = np.diff(y_test) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            else:
                directional_accuracy = 0.0
        
        model_comparison.append({
            'Model': model_name.replace('_', ' ').title(),
            'R²': f"{performance['r2']:.3f}",
            'MAE': f"${performance['mae']:.2f}",
            'RMSE': f"${performance['rmse']:.2f}",
            'MAPE': f"{performance['mape']:.1f}%",
            'Directional Accuracy': f"{directional_accuracy:.1f}%"
        })
    
    comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(comparison_df, width='stretch')

def display_rolling_backtest_results(results):
    """Display rolling backtest (walk-forward validation) results."""
    
    backtest_results = results.get('rolling_backtest_results', {})
    
    if not backtest_results or not backtest_results.get('model_performance'):
        st.warning("No rolling backtest results available.")
        return
    
    # Display backtest summary
    st.subheader("📊 Backtest Summary")
    
    fold_info = backtest_results.get('fold_info', [])
    if fold_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Folds", len(fold_info))
        with col2:
            if fold_info:
                train_days = fold_info[0]['train_end'] - fold_info[0]['train_start']
                st.metric("Training Window", f"{train_days} days")
        with col3:
            if fold_info:
                test_days = fold_info[0]['test_end'] - fold_info[0]['test_start']
                st.metric("Test Window", f"{test_days} days")
    
    # Model performance comparison across folds
    st.subheader("📈 Model Performance Across Folds")
    
    model_performance = backtest_results.get('model_performance', {})
    
    if model_performance:
        # Create performance summary table
        performance_data = []
        
        for model_name, model_data in model_performance.items():
            metrics = model_data.get('metrics', [])
            if metrics:
                # Calculate average metrics across all folds
                avg_r2 = np.mean([m['r2'] for m in metrics])
                avg_mae = np.mean([m['mae'] for m in metrics])
                avg_rmse = np.mean([m['rmse'] for m in metrics])
                avg_mape = np.mean([m['mape'] for m in metrics])
                
                # Calculate standard deviation for stability
                std_r2 = np.std([m['r2'] for m in metrics])
                std_mae = np.std([m['mae'] for m in metrics])
                std_rmse = np.std([m['rmse'] for m in metrics])
                std_mape = np.std([m['mape'] for m in metrics])
                
                performance_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Avg R²': f"{avg_r2:.3f} ± {std_r2:.3f}",
                    'Avg MAE': f"{avg_mae:.3f} ± {std_mae:.3f}",
                    'Avg RMSE': f"{avg_rmse:.3f} ± {std_rmse:.3f}",
                    'Avg MAPE': f"{avg_mape:.1f}% ± {std_mape:.1f}%",
                    'Folds': len(metrics)
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, width='stretch')
            
            # Find best performing model
            best_model = max(performance_data, key=lambda x: float(x['Avg R²'].split(' ± ')[0]))
            st.success(f"🏆 **Best Model**: {best_model['Model']} (R² = {best_model['Avg R²']})")
    
    # Performance stability analysis
    st.subheader("📊 Performance Stability Analysis")
    
    if model_performance:
        # Create stability chart
        stability_data = []
        
        for model_name, model_data in model_performance.items():
            metrics = model_data.get('metrics', [])
            if metrics:
                for metric in metrics:
                    stability_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Fold': metric['fold'],
                        'R²': metric['r2'],
                        'MAE': metric['mae'],
                        'RMSE': metric['rmse'],
                        'MAPE': metric['mape']
                    })
        
        if stability_data:
            stability_df = pd.DataFrame(stability_data)
            
            # R² stability chart
            fig_r2 = px.line(
                stability_df, 
                x='Fold', 
                y='R²', 
                color='Model',
                title='R² Score Stability Across Folds',
                markers=True
            )
            fig_r2.update_layout(height=400)
            st.plotly_chart(fig_r2, use_container_width=True)
            
            # MAE stability chart
            fig_mae = px.line(
                stability_df, 
                x='Fold', 
                y='MAE', 
                color='Model',
                title='MAE Stability Across Folds',
                markers=True
            )
            fig_mae.update_layout(height=400)
            st.plotly_chart(fig_mae, use_container_width=True)
    
    # Fold-by-fold analysis
    st.subheader("📋 Fold-by-Fold Analysis")
    
    if fold_info:
        # Create fold summary table
        fold_data = []
        
        for fold in fold_info:
            fold_data.append({
                'Fold': fold['fold'],
                'Train Period': f"{fold['train_dates'][0].strftime('%Y-%m-%d')} to {fold['train_dates'][1].strftime('%Y-%m-%d')}",
                'Test Period': f"{fold['test_dates'][0].strftime('%Y-%m-%d')} to {fold['test_dates'][1].strftime('%Y-%m-%d')}",
                'Train Days': fold['train_end'] - fold['train_start'],
                'Test Days': fold['test_end'] - fold['test_start']
            })
        
        fold_df = pd.DataFrame(fold_data)
        st.dataframe(fold_df, width='stretch')
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    if model_performance and fold_info:
        # Calculate insights
        total_folds = len(fold_info)
        
        # Find most stable model (lowest std deviation in R²)
        model_stability = {}
        for model_name, model_data in model_performance.items():
            metrics = model_data.get('metrics', [])
            if metrics:
                r2_values = [m['r2'] for m in metrics]
                model_stability[model_name] = np.std(r2_values)
        
        if model_stability:
            most_stable = min(model_stability.keys(), key=lambda x: model_stability[x])
            st.info(f"🎯 **Most Stable Model**: {most_stable.replace('_', ' ').title()} (R² std: {model_stability[most_stable]:.3f})")
        
        # Performance consistency
        st.markdown(f"""
        **Rolling Backtest Insights:**
        - **Total Folds**: {total_folds} time periods tested
        - **Validation Method**: Walk-forward validation (no data leakage)
        - **Target**: Returns-based prediction (better stationarity)
        - **Evaluation**: Each model tested on {fold_info[0]['test_end'] - fold_info[0]['test_start']} days of unseen data per fold
        
        **Benefits of Rolling Backtest:**
        - ✅ Tests model performance across different market conditions
        - ✅ Provides more robust evaluation than single train/test split
        - ✅ Shows model stability over time
        - ✅ Identifies models that work consistently across periods
        """)

def display_investment_results(results):
    """Display investment calculator results."""
    
    investment_results = results.get('investment_results', {})
    investment_amount = results.get('investment_amount', 10000)
    investment_days = results.get('investment_days', 30)
    
    if not investment_results:
        st.warning("No investment results available.")
        return
    
    # Display investment parameters
    st.subheader("📊 Investment Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Investment Amount", f"${investment_amount:,.2f}")
    with col2:
        st.metric("Investment Period", f"{investment_days} days")
    with col3:
        st.metric("Stock Symbol", results.get('symbol', 'N/A'))
    
    # Display strategy results
    st.subheader("📈 Strategy Comparison")
    
    # Create results table
    strategy_data = []
    
    for strategy_key, strategy_data_dict in investment_results.items():
        if strategy_data_dict:
            strategy_data.append({
                'Strategy': strategy_data_dict['strategy'],
                'Initial Value': f"${strategy_data_dict['initial_value']:,.2f}",
                'Final Value': f"${strategy_data_dict['final_value']:,.2f}",
                'Return Amount': f"${strategy_data_dict['return_amount']:,.2f}",
                'Return %': f"{strategy_data_dict['return_percentage']:+.2f}%"
            })
    
    if strategy_data:
        strategy_df = pd.DataFrame(strategy_data)
        st.dataframe(strategy_df, width='stretch')
        
        # Visual comparison
        st.subheader("📊 Return Comparison Chart")
        
        strategies = [row['Strategy'] for row in strategy_data]
        returns = [float(row['Return %'].replace('%', '').replace('+', '')) for row in strategy_data]
        
        fig = go.Figure(data=[
            go.Bar(
                x=strategies,
                y=returns,
                marker_color=['green' if r > 0 else 'red' for r in returns],
                text=[f"{r:+.1f}%" for r in returns],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Investment Returns Comparison ({investment_days} days)",
            xaxis_title="Strategy",
            yaxis_title="Return (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy insights
        st.subheader("💡 Strategy Insights")
        
        # Find best and worst strategies
        best_strategy = max(strategy_data, key=lambda x: float(x['Return %'].replace('%', '').replace('+', '')))
        worst_strategy = min(strategy_data, key=lambda x: float(x['Return %'].replace('%', '').replace('+', '')))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🏆 **Best Strategy**: {best_strategy['Strategy']}")
            st.write(f"Return: {best_strategy['Return %']}")
            st.write(f"Final Value: {best_strategy['Final Value']}")
        
        with col2:
            if worst_strategy['Strategy'] != best_strategy['Strategy']:
                st.warning(f"⚠️ **Worst Strategy**: {worst_strategy['Strategy']}")
                st.write(f"Return: {worst_strategy['Return %']}")
                st.write(f"Final Value: {worst_strategy['Final Value']}")
        
        # Model performance vs buy & hold
        if 'buy_hold' in investment_results and 'model_trading' in investment_results:
            buy_hold_return = investment_results['buy_hold']['return_percentage']
            model_return = investment_results['model_trading']['return_percentage']
            
            if model_return > buy_hold_return:
                improvement = model_return - buy_hold_return
                st.success(f"🎯 **Model Outperformed Buy & Hold by {improvement:.2f}%**")
            else:
                underperformance = buy_hold_return - model_return
                st.info(f"📉 **Model Underperformed Buy & Hold by {underperformance:.2f}%**")
        
        # Risk assessment
        st.subheader("⚠️ Risk Assessment")
        st.markdown("""
        **Important Disclaimers:**
        - These are **backtested results** based on historical data
        - Past performance does not guarantee future results
        - Real trading involves transaction costs, slippage, and market impact
        - Model predictions may not be accurate in real-time trading
        - Consider your risk tolerance before making investment decisions
        """)
        
        # Annualized returns
        if investment_days > 0:
            st.subheader("📅 Annualized Returns")
            annualized_data = []
            
            for strategy_key, strategy_data_dict in investment_results.items():
                if strategy_data_dict:
                    daily_return = strategy_data_dict['return_percentage'] / investment_days
                    annualized_return = daily_return * 365
                    
                    annualized_data.append({
                        'Strategy': strategy_data_dict['strategy'],
                        'Annualized Return': f"{annualized_return:+.2f}%"
                    })
            
            if annualized_data:
                annualized_df = pd.DataFrame(annualized_data)
                st.dataframe(annualized_df, width='stretch')

def display_analysis_explanation():
    """Display explanation of the improved analysis."""
    
    st.markdown("""
    ## 🔍 How the Enhanced System Works
    
    ### ✅ **Key Improvements:**
    
    1. **No Data Leakage**: Features only use past information, never future data
    2. **Time Series Split**: Training on earlier data, testing on later data (realistic)
    3. **Realistic Constraints**: Predictions bounded by market limits (max 10% daily change)
    4. **Proper Feature Engineering**: Uses lookback window approach instead of complex lag features
    5. **Earnings Integration**: Incorporates earnings data and surprise analysis
    6. **Realistic Performance**: R² scores typically 70-90% (not artificially inflated)
    
    ### 📊 **Feature Engineering:**
    - **Past {lookback_window} days**: Open, High, Low, Close, Volume
    - **Technical indicators**: Moving averages, momentum, volatility
    - **Earnings features**: Days since/until earnings, surprise history, impact analysis
    - **Total features**: {lookback_window}×5 + 6 indicators + 6 earnings = {lookback_window*5 + 12} features
    
    ### 💰 **Earnings Features:**
    1. **Days since last earnings**: Time since most recent earnings announcement
    2. **Days until next earnings**: Time until upcoming earnings announcement
    3. **Last earnings surprise**: Most recent earnings surprise percentage
    4. **Last earnings impact**: Price impact of most recent earnings
    5. **Average surprise**: Average earnings surprise over last 4 quarters
    6. **Earnings volatility**: Standard deviation of recent earnings surprises
    
    ### 🎯 **Earnings Analysis:**
    - **Earnings Events**: Marked on price charts with color-coded surprise indicators
    - **Impact Analysis**: Shows correlation between earnings surprises and price movements
    - **Tooltips**: Hover over earnings markers to see detailed earnings data
    - **Beat/Miss Tracking**: Visual indicators for earnings beats, misses, and meets
    
    """)

if __name__ == "__main__":
    main()
