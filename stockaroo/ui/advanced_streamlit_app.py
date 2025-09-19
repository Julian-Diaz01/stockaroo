"""
Advanced Streamlit Web UI with Cross-Market Analysis.
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
    page_title="Advanced Stock Analytics Dashboard",
    page_icon="🌍",
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
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Advanced Stock Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Analysis type selection
    analysis_type = st.radio(
        "Select Analysis Type:",
        ["Single Stock Analysis", "Model Management"],
        horizontal=True
    )
    
    if analysis_type == "Single Stock Analysis":
        single_stock_analysis()
    elif analysis_type == "Model Management":
        model_management()

def single_stock_analysis():
    """Single stock analysis interface."""
    st.header("📈 Single Stock Analysis")
    
    # Sidebar for parameters
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        
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
        run_analysis_btn = st.button("🚀 Run Single Stock Analysis", type="primary", use_container_width=True)
        
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
        with st.spinner(f"Running single stock analysis for {symbol}..."):
            try:
                results = run_single_stock_analysis(
                    symbol, period_value, interval, 
                    models_to_use, lookback_window, test_size
                )
                st.session_state.analysis_results = results
                
                # Success message
                st.markdown("""
                <div class="success-message">
                    ✅ Single stock analysis completed successfully!
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return
    
    # Display results if available
    if st.session_state.analysis_results:
        display_single_stock_results(st.session_state.analysis_results)


def model_management():
    """Model management interface."""
    st.header("🔧 Model Management")
    
    # Show saved models
    st.subheader("📁 Saved Models")
    
    # Single stock models
    st.write("**Single Stock Models:**")
    try:
        predictor = StockPredictor()
        saved_models = predictor.list_saved_models()
        
        if saved_models:
            for model in saved_models:
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.write(f"📄 {model['filename']}")
                with col2:
                    st.write(f"Model: {model['model_name']}")
                with col3:
                    st.write(f"Symbol: {model['symbol']}")
                with col4:
                    if st.button("🗑️", key=f"delete_{model['filename']}", help="Delete model"):
                        import os
                        os.remove(model['filepath'])
                        st.rerun()
        else:
            st.info("No single stock models saved yet.")
    except Exception as e:
        st.error(f"Error loading single stock models: {e}")
    
    
    # Model performance comparison
    st.subheader("📊 Model Performance Comparison")
    
    if st.button("🔄 Refresh Model Performance"):
        try:
            # Quick performance test
            predictor = StockPredictor()
            collector = StockDataCollector("AAPL")
            data = collector.get_stock_data(period="6mo")
            
            X, y = predictor.prepare_time_series_data(data, lookback_window=10)
            X_train, X_test, y_train, y_test = predictor.time_series_split(X, y)
            
            performance_data = []
            for model_name in ['linear_regression', 'ridge', 'lasso']:
                predictor.train_model(model_name, X_train, y_train)
                results = predictor.evaluate_model(model_name, X_test, y_test)
                performance_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'R² Score': results['r2'],
                    'RMSE': results['rmse'],
                    'MAPE': results['mape']
                })
            
            df = pd.DataFrame(performance_data)
            st.dataframe(df.style.format({
                'R² Score': '{:.4f}',
                'RMSE': '{:.2f}',
                'MAPE': '{:.2f}%'
            }), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error running performance test: {e}")

def run_single_stock_analysis(symbol, period, interval, models_to_use, lookback_window, test_size):
    """Run single stock analysis."""
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


def display_single_stock_results(results):
    """Display single stock analysis results."""
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
    st.header("🎯 Predictions vs Actual")
    display_predictions_comparison(results)
    
    # Next day prediction
    if results['next_day_pred'] is not None:
        st.header("🔮 Next Day Prediction")
        display_next_day_prediction(results)


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
    """Display model performance metrics."""
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
        title='Model Performance Comparison',
        color='R² Score',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def display_predictions_comparison(results):
    """Display predictions vs actual values."""
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
        title='Predictions vs Actual Values',
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

if __name__ == "__main__":
    main()
        current_price = market_data['america'].iloc[-1]['Close']
    else:
        current_price = 6500  # Default fallback
    
    next_day_pred = results['next_day_pred']
    change_pct = ((next_day_pred - current_price) / current_price) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current S&P 500", f"{current_price:.2f}")
    with col2:
        st.metric("Predicted Tomorrow", f"{next_day_pred:.2f}")
    with col3:
        st.metric("Expected Change", f"{change_pct:+.2f}%")
    
    # Prediction explanation
    st.markdown("""
    <div class="info-message">
        <strong>Cross-Market Prediction:</strong> This prediction uses Hong Kong and European market data to forecast the US market, bounded by realistic constraints (max 5% daily change).
    </div>
    """, unsafe_allow_html=True)

def display_ad_analysis(ad_analysis):
    """Display accumulation/distribution analysis."""
    if not ad_analysis:
        st.warning("No A/D analysis available.")
        return
    
    # Create A/D analysis dataframe
    ad_data = []
    for market, analysis in ad_analysis.items():
        ad_data.append({
            'Market': market.replace('_', ' ').title(),
            'A/D Trend': analysis.get('ad_trend', 'unknown'),
            'A/D Momentum': f"{analysis.get('ad_momentum', 0):.0f}" if not pd.isna(analysis.get('ad_momentum', 0)) else "N/A",
            'Oscillator Trend': analysis.get('oscillator_trend', 'unknown'),
            'Volume Trend': analysis.get('volume_trend', 'unknown')
        })
    
    df = pd.DataFrame(ad_data)
    
    # Color code the trends
    def color_trend(val):
        if val == 'bullish':
            return 'background-color: #d4edda'
        elif val == 'bearish':
            return 'background-color: #f8d7da'
        elif val == 'positive':
            return 'background-color: #d4edda'
        elif val == 'negative':
            return 'background-color: #f8d7da'
        elif val == 'increasing':
            return 'background-color: #d4edda'
        elif val == 'decreasing':
            return 'background-color: #f8d7da'
        return ''
    
    st.dataframe(
        df.style.applymap(color_trend, subset=['A/D Trend', 'Oscillator Trend', 'Volume Trend']),
        use_container_width=True
    )
    
    # A/D explanation
    st.markdown("""
    <div class="info-message">
        <strong>Accumulation/Distribution Analysis:</strong>
        <ul>
            <li><strong>Bullish/Bearish:</strong> Overall money flow trend</li>
            <li><strong>Positive/Negative:</strong> Oscillator direction</li>
            <li><strong>Increasing/Decreasing:</strong> Volume trend</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_cross_market_correlations(full_data):
    """Display cross-market correlations."""
    # Calculate correlations
    correlation_data = []
    
    if 'hong_kong_returns' in full_data.columns and 'europe_returns' in full_data.columns:
        hk_eu_corr = full_data['hong_kong_returns'].corr(full_data['europe_returns'])
        correlation_data.append({
            'Markets': 'Hong Kong ↔ Europe',
            'Correlation': hk_eu_corr,
            'Strength': 'Strong' if abs(hk_eu_corr) > 0.7 else 'Moderate' if abs(hk_eu_corr) > 0.3 else 'Weak'
        })
    
    if 'europe_returns' in full_data.columns and 'america_returns' in full_data.columns:
        eu_us_corr = full_data['europe_returns'].corr(full_data['america_returns'])
        correlation_data.append({
            'Markets': 'Europe ↔ America',
            'Correlation': eu_us_corr,
            'Strength': 'Strong' if abs(eu_us_corr) > 0.7 else 'Moderate' if abs(eu_us_corr) > 0.3 else 'Weak'
        })
    
    if 'hong_kong_returns' in full_data.columns and 'america_returns' in full_data.columns:
        hk_us_corr = full_data['hong_kong_returns'].corr(full_data['america_returns'])
        correlation_data.append({
            'Markets': 'Hong Kong ↔ America',
            'Correlation': hk_us_corr,
            'Strength': 'Strong' if abs(hk_us_corr) > 0.7 else 'Moderate' if abs(hk_us_corr) > 0.3 else 'Weak'
        })
    
    if correlation_data:
        df = pd.DataFrame(correlation_data)
        
        # Color code correlations
        def color_correlation(val):
            if abs(val) > 0.7:
                return 'background-color: #d4edda' if val > 0 else 'background-color: #f8d7da'
            elif abs(val) > 0.3:
                return 'background-color: #fff3cd'
            return 'background-color: #f8f9fa'
        
        st.dataframe(
            df.style.applymap(color_correlation, subset=['Correlation']).format({
                'Correlation': '{:.3f}'
            }),
            use_container_width=True
        )
        
        # Correlation chart
        fig = px.bar(
            df,
            x='Markets',
            y='Correlation',
            title='Cross-Market Correlations',
            color='Correlation',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No correlation data available.")

def display_cross_market_indices_chart(results):
    """Display cross-market indices chart with prediction."""
    try:
        # Get market data for the chart
        collector = MultiMarketCollector()
        market_data = collector.get_all_markets_data(period=results['period'])
        
        if not market_data:
            st.warning("No market data available for chart.")
            return
        
        # Create the chart
        fig = go.Figure()
        
        # Add each market index
        colors = {
            'hong_kong': '#FF6B6B',  # Red
            'europe': '#4ECDC4',     # Teal
            'america': '#45B7D1'     # Blue
        }
        
        market_names = {
            'hong_kong': 'Hong Kong (Hang Seng)',
            'europe': 'Europe (STOXX 600)',
            'america': 'America (S&P 500)'
        }
        
        # Plot each market's close prices
        for market, data in market_data.items():
            if market in colors and len(data) > 0:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=market_names[market],
                    line=dict(color=colors[market], width=2),
                    hovertemplate=f'<b>{market_names[market]}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Price: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
        
        # Add prediction point if available
        if results['next_day_pred'] is not None:
            # Get the last date from America data
            if 'america' in market_data and len(market_data['america']) > 0:
                last_date = market_data['america'].index[-1]
                next_date = last_date + pd.Timedelta(days=1)
                
                fig.add_trace(go.Scatter(
                    x=[next_date],
                    y=[results['next_day_pred']],
                    mode='markers',
                    name='Predicted S&P 500',
                    marker=dict(
                        color='#FFD700',  # Gold
                        size=12,
                        symbol='diamond',
                        line=dict(color='black', width=2)
                    ),
                    hovertemplate='<b>Predicted S&P 500</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Predicted Price: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title='Cross-Market Indices with Prediction',
            xaxis_title='Date',
            yaxis_title='Index Value',
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class="info-message">
            <strong>Chart Explanation:</strong>
            <ul>
                <li><strong>Red Line:</strong> Hong Kong Hang Seng Index</li>
                <li><strong>Teal Line:</strong> European STOXX 600 Index</li>
                <li><strong>Blue Line:</strong> American S&P 500 Index</li>
                <li><strong>Gold Diamond:</strong> Predicted next-day S&P 500 value</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error creating cross-market chart: {str(e)}")

if __name__ == "__main__":
    main()
