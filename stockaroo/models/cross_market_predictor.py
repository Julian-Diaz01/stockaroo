"""
Cross-market predictor for multi-market analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import logging
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ..data.multi_market_collector import MultiMarketCollector

logger = logging.getLogger(__name__)

class CrossMarketPredictor:
    """
    Predictor for cross-market analysis using Hong Kong, European, and American indices.
    """
    
    def __init__(self):
        """Initialize the cross-market predictor."""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.model_scores = {}
        self.feature_names = None
        self.data_collector = MultiMarketCollector()
        self.models_dir = "saved_models/cross_market"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_cross_market_data(self, period: str = "1y", interval: str = "1d", 
                                 lookback_window: int = 5) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare data for cross-market prediction.
        
        Args:
            period: Data period
            interval: Data interval
            lookback_window: Number of past days to include as features
            
        Returns:
            Tuple of (features, target, full_data)
        """
        logger.info("Preparing cross-market data...")
        
        # Get cross-market data
        features, target, full_data = self.data_collector.prepare_cross_market_prediction_data(period, interval)
        
        # Add lookback features
        enhanced_features = []
        enhanced_targets = []
        
        for i in range(lookback_window, len(features)):
            # Create feature vector with lookback
            feature_row = []
            
            # Current day features
            feature_row.extend(features[i])
            
            # Previous days features (lookback)
            for j in range(1, lookback_window + 1):
                if i - j >= 0:
                    feature_row.extend(features[i - j])
                else:
                    # Pad with zeros if not enough history
                    feature_row.extend([0] * features.shape[1])
            
            enhanced_features.append(feature_row)
            enhanced_targets.append(target[i])
        
        enhanced_features = np.array(enhanced_features)
        enhanced_targets = np.array(enhanced_targets)
        
        # Handle NaN values
        if len(enhanced_features) > 0:
            nan_mask = np.isnan(enhanced_features).any(axis=1) | np.isnan(enhanced_targets)
            enhanced_features = enhanced_features[~nan_mask]
            enhanced_targets = enhanced_targets[~nan_mask]
        else:
            raise ValueError("No valid data samples after feature engineering")
        
        logger.info(f"Enhanced cross-market data: {enhanced_features.shape[0]} samples, {enhanced_features.shape[1]} features")
        
        return enhanced_features, enhanced_targets, full_data
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        logger.info(f"Training cross-market {model_name} model")
        
        # Create a fresh model instance
        if model_name == 'linear_regression':
            model = LinearRegression()
        elif model_name == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_name == 'lasso':
            model = Lasso(alpha=0.1)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        # Calculate training score
        train_score = model.score(X_train, y_train)
        
        logger.info(f"Cross-market {model_name} training completed. R² score: {train_score:.4f}")
        
        return {
            'model_name': model_name,
            'train_score': train_score,
            'model': model
        }
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': y_pred
        }
        
        self.model_scores[model_name] = metrics
        
        logger.info(f"Cross-market {model_name} evaluation completed:")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def time_series_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
        """Split data using time series split."""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Cross-market time series split: {len(X_train)} training, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def predict_next_day(self, model_name: str, lookback_window: int = 2) -> float:
        """
        Predict next day's US market using current market data.
        
        Args:
            model_name: Name of the trained model
            lookback_window: Number of past days to use (must match training)
            
        Returns:
            Predicted US market close for next day
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Get latest market data
        market_data = self.data_collector.get_all_markets_data(period="1mo", interval="1d")
        aligned_data = self.data_collector.align_market_data(market_data)
        features_data = self.data_collector.calculate_cross_market_features(aligned_data)
        
        # Prepare features exactly like in training
        feature_cols = [col for col in features_data.columns if not col.endswith('_Close')]
        features = features_data[feature_cols].values
        
        # Create feature vector with lookback (same as training)
        if len(features) >= lookback_window:
            # Get the most recent features
            recent_features = features[-1]
            
            # Create feature vector with lookback
            feature_row = list(recent_features)
            
            # Add lookback features (pad with zeros if not enough history)
            for j in range(1, lookback_window + 1):
                if len(features) > j:
                    feature_row.extend(features[-1-j])
                else:
                    feature_row.extend([0] * len(recent_features))
            
            # Make prediction
            features_array = np.array([feature_row])
            prediction = model.predict(features_array)[0]
            
            # Apply reasonable constraints (max 5% change for index)
            current_us_close = aligned_data['america_Close'].iloc[-1]
            max_change = 0.05  # 5% max change per day for index
            min_price = current_us_close * (1 - max_change)
            max_price = current_us_close * (1 + max_change)
            
            prediction = np.clip(prediction, min_price, max_price)
            
            return prediction
        else:
            raise ValueError(f"Not enough data for prediction. Need at least {lookback_window} days.")
    
    def analyze_accumulation_distribution(self, period: str = "3mo") -> Dict[str, pd.DataFrame]:
        """
        Analyze accumulation/distribution patterns across markets.
        
        Args:
            period: Data period for analysis
            
        Returns:
            Dictionary with A/D analysis for each market
        """
        logger.info("Analyzing accumulation/distribution patterns...")
        
        # Get market data
        market_data = self.data_collector.get_all_markets_data(period, "1d")
        
        # Calculate A/D indicators
        ad_data = self.data_collector.get_accumulation_distribution_data(market_data)
        
        # Analyze patterns
        analysis = {}
        
        for market in self.data_collector.market_indices.keys():
            market_analysis = {}
            
            # Get A/D data for this market
            ad_line_col = f"{market}_ad_line"
            ad_osc_col = f"{market}_ad_oscillator"
            
            if ad_line_col in ad_data.columns:
                ad_line = ad_data[ad_line_col]
                ad_osc = ad_data[ad_osc_col] if ad_osc_col in ad_data.columns else None
                
                # Calculate trends
                market_analysis['ad_trend'] = 'bullish' if ad_line.iloc[-1] > ad_line.iloc[-5] else 'bearish'
                market_analysis['ad_momentum'] = ad_line.iloc[-1] - ad_line.iloc[-10] if len(ad_line) >= 10 else 0
                
                if ad_osc is not None:
                    market_analysis['oscillator_trend'] = 'positive' if ad_osc.iloc[-1] > 0 else 'negative'
                    market_analysis['oscillator_momentum'] = ad_osc.iloc[-1] - ad_osc.iloc[-5] if len(ad_osc) >= 5 else 0
                
                # Volume analysis
                if f"{market}_Volume" in ad_data.columns:
                    volume = ad_data[f"{market}_Volume"]
                    market_analysis['volume_trend'] = 'increasing' if volume.iloc[-1] > volume.iloc[-5] else 'decreasing'
                
                analysis[market] = market_analysis
        
        return analysis
    
    def save_model(self, model_name: str, additional_info: dict = None) -> str:
        """Save a trained cross-market model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cross_market_{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        model_data = {
            'model': self.trained_models[model_name],
            'model_name': model_name,
            'type': 'cross_market',
            'timestamp': timestamp,
            'scores': self.model_scores.get(model_name, {}),
            'additional_info': additional_info or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Cross-market model {model_name} saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> dict:
        """Load a saved cross-market model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model_name = model_data['model_name']
        self.trained_models[model_name] = model_data['model']
        self.model_scores[model_name] = model_data.get('scores', {})
        
        logger.info(f"Cross-market model {model_name} loaded from {filepath}")
        return model_data

def demonstrate_cross_market_prediction():
    """Demonstrate cross-market prediction system."""
    logger.info("Demonstrating cross-market prediction system...")
    
    # Initialize predictor
    predictor = CrossMarketPredictor()
    
    # Prepare data
    X, y, full_data = predictor.prepare_cross_market_data(period="6mo", lookback_window=2)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.time_series_split(X, y, test_size=0.2)
    
    # Train model
    predictor.train_model('linear_regression', X_train, y_train)
    
    # Evaluate model
    results = predictor.evaluate_model('linear_regression', X_test, y_test)
    
    print(f"Cross-market prediction results:")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"MAPE: {results['mape']:.2f}%")
    
    # Next day prediction
    next_day_pred = predictor.predict_next_day('linear_regression')
    print(f"Predicted next day US market: {next_day_pred:.2f}")
    
    # A/D analysis
    ad_analysis = predictor.analyze_accumulation_distribution()
    print(f"\nAccumulation/Distribution Analysis:")
    for market, analysis in ad_analysis.items():
        print(f"{market}: {analysis}")
    
    return predictor, results

if __name__ == "__main__":
    demonstrate_cross_market_prediction()
