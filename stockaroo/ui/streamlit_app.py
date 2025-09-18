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
warnings.filterwarnings('ignore')

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
        models_to_use = st.multiselect(
            "Select Models to Compare",
            ["Linear Regression", "Ridge", "Lasso", "Random Forest"],
            default=["Linear Regression", "Lasso"]
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
                    models_to_use, lookback_window, test_size
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

def run_analysis(symbol, period, interval, models_to_use, lookback_window, test_size):
    """Run the improved stock analysis pipeline."""
    
    # Step 1: Data Collection
    collector = StockDataCollector(symbol)
    raw_data = collector.get_stock_data(period=period, interval=interval)
    company_info = collector.get_company_info()
    
    # Step 2: Prepare data using improved method
    predictor = StockPredictor()
    X, y = predictor.prepare_time_series_data(
        raw_data, 
        lookback_window=lookback_window, 
        prediction_horizon=1
    )
    
    # Step 3: Split data temporally
    X_train, X_test, y_train, y_test = predictor.time_series_split(X, y, test_size)
    
    # Step 4: Train and evaluate models
    model_mapping = {
        "Linear Regression": "linear_regression",
        "Ridge": "ridge", 
        "Lasso": "lasso",
        "Random Forest": "random_forest"
    }
    
    selected_models = [model_mapping[model] for model in models_to_use if model in model_mapping]
    
    eval_results = {}
    for model_name in selected_models:
        try:
            predictor.train_model(model_name, X_train, y_train)
            eval_results[model_name] = predictor.evaluate_model(
                model_name, X_test, y_test
            )
        except Exception as e:
            st.warning(f"Error with {model_name}: {str(e)}")
    
    # Get best model
    best_model = max(eval_results.keys(), key=lambda x: eval_results[x]['r2']) if eval_results else None
    
    # Next day prediction
    next_day_pred = None
    if best_model:
        try:
            recent_data = raw_data.tail(30)  # Use last 30 days
            next_day_pred = predictor.predict_next_day(best_model, recent_data, lookback_window)
        except Exception as e:
            st.warning(f"Could not make next day prediction: {e}")
    
    return {
        'raw_data': raw_data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'company_info': company_info,
        'eval_results': eval_results,
        'best_model': best_model,
        'next_day_pred': next_day_pred,
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'lookback_window': lookback_window
    }

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
    
    # Price chart
    st.header("📈 Stock Price Chart")
    create_price_chart(results['raw_data'], results['symbol'])
    
    # Model performance
    st.header("🤖 Model Performance")
    display_model_performance(results['eval_results'], results['best_model'])
    
    # Predictions vs Actual
    st.header("🎯 Predictions vs Actual (Time Series Split)")
    display_predictions_comparison(results)
    
    # Next day prediction
    if results['next_day_pred'] is not None:
        st.header("🔮 Next Day Prediction (Realistic)")
        display_next_day_prediction(results)
    
    # Analysis explanation
    st.header("📚 Analysis Explanation")
    display_analysis_explanation()

def create_price_chart(data, symbol):
    """Create interactive price chart."""
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ))
    
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True
    )
    
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
        use_container_width=True
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
        use_container_width=True
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

def display_analysis_explanation():
    """Display explanation of the improved analysis."""
    
    st.markdown("""
    ## 🔍 How the Improved System Works
    
    ### ✅ **Key Improvements:**
    
    1. **No Data Leakage**: Features only use past information, never future data
    2. **Time Series Split**: Training on earlier data, testing on later data (realistic)
    3. **Realistic Constraints**: Predictions bounded by market limits (max 10% daily change)
    4. **Proper Feature Engineering**: Uses lookback window approach instead of complex lag features
    5. **Realistic Performance**: R² scores typically 70-90% (not artificially inflated)
    
    ### 📊 **Feature Engineering:**
    - **Past {lookback_window} days**: Open, High, Low, Close, Volume
    - **Technical indicators**: Moving averages, momentum, volatility
    - **Total features**: {lookback_window}×5 + 6 indicators = {lookback_window*5 + 6} features
    
    """)

if __name__ == "__main__":
    main()
