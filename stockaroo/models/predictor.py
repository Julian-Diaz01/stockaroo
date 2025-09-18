"""
Improved Stock Predictor with proper time series handling.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import logging
import pickle
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class StockPredictor:
    """
    Improved stock predictor with proper time series handling.
    """
    
    def __init__(self):
        """Initialize the predictor with default models."""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.model_scores = {}
        self.feature_names = None
        self.scaler = None
        self.models_dir = "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_time_series_data(self, data: pd.DataFrame, target_col: str = 'Close', 
                                lookback_window: int = 10, prediction_horizon: int = 1) -> tuple:
        """
        Prepare data for time series prediction with proper feature engineering.
        
        Args:
            data: Stock data
            target_col: Target column
            lookback_window: Number of past days to use as features
            prediction_horizon: Days ahead to predict
            
        Returns:
            X, y: Features and targets
        """
        df = data.copy()
        
        # Create features using only past information
        features = []
        targets = []
        
        for i in range(lookback_window, len(df) - prediction_horizon + 1):
            # Features: past lookback_window days
            feature_row = []
            
            # Price features (past days)
            for j in range(lookback_window):
                idx = i - lookback_window + j
                feature_row.extend([
                    df.iloc[idx]['Open'],
                    df.iloc[idx]['High'], 
                    df.iloc[idx]['Low'],
                    df.iloc[idx]['Close'],
                    df.iloc[idx]['Volume']
                ])
            
            # Technical indicators (calculated from past data only)
            if i >= 20:  # Ensure we have enough data for indicators
                # Simple moving averages
                ma_5 = df.iloc[i-5:i]['Close'].mean()
                ma_10 = df.iloc[i-10:i]['Close'].mean()
                ma_20 = df.iloc[i-20:i]['Close'].mean()
                
                # Price momentum
                momentum_5 = (df.iloc[i-1]['Close'] / df.iloc[i-6]['Close'] - 1) if i >= 6 else 0
                momentum_10 = (df.iloc[i-1]['Close'] / df.iloc[i-11]['Close'] - 1) if i >= 11 else 0
                
                # Volatility (rolling std of returns)
                if i >= 10:
                    returns = df.iloc[i-10:i]['Close'].pct_change().dropna()
                    volatility = returns.std() if len(returns) > 1 else 0
                else:
                    volatility = 0
                
                feature_row.extend([ma_5, ma_10, ma_20, momentum_5, momentum_10, volatility])
            else:
                # Pad with zeros if not enough data
                feature_row.extend([0, 0, 0, 0, 0, 0])
            
            features.append(feature_row)
            
            # Target: future price
            target = df.iloc[i + prediction_horizon - 1][target_col]
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        logger.info(f"Training {model_name} model")
        
        # Create a fresh model instance to avoid conflicts
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
        
        logger.info(f"{model_name} training completed. R² score: {train_score:.4f}")
        
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
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred
        }
        
        self.model_scores[model_name] = metrics
        
        logger.info(f"{model_name} evaluation completed:")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def predict_next_day(self, model_name: str, recent_data: pd.DataFrame, 
                        lookback_window: int = 10) -> float:
        """
        Predict the next day's price using the most recent data.
        
        Args:
            model_name: Name of the trained model
            recent_data: Recent stock data (last lookback_window + 20 days)
            lookback_window: Number of past days to use as features
            
        Returns:
            Predicted price for the next day
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        
        # Prepare features from recent data
        feature_row = []
        
        # Price features (past lookback_window days)
        for j in range(lookback_window):
            idx = len(recent_data) - lookback_window + j
            feature_row.extend([
                recent_data.iloc[idx]['Open'],
                recent_data.iloc[idx]['High'],
                recent_data.iloc[idx]['Low'], 
                recent_data.iloc[idx]['Close'],
                recent_data.iloc[idx]['Volume']
            ])
        
        # Technical indicators
        if len(recent_data) >= 20:
            # Simple moving averages
            ma_5 = recent_data.iloc[-5:]['Close'].mean()
            ma_10 = recent_data.iloc[-10:]['Close'].mean()
            ma_20 = recent_data.iloc[-20:]['Close'].mean()
            
            # Price momentum
            momentum_5 = (recent_data.iloc[-1]['Close'] / recent_data.iloc[-6]['Close'] - 1) if len(recent_data) >= 6 else 0
            momentum_10 = (recent_data.iloc[-1]['Close'] / recent_data.iloc[-11]['Close'] - 1) if len(recent_data) >= 11 else 0
            
            # Volatility
            if len(recent_data) >= 10:
                returns = recent_data.iloc[-10:]['Close'].pct_change().dropna()
                volatility = returns.std() if len(returns) > 1 else 0
            else:
                volatility = 0
                
            feature_row.extend([ma_5, ma_10, ma_20, momentum_5, momentum_10, volatility])
        else:
            feature_row.extend([0, 0, 0, 0, 0, 0])
        
        # Make prediction
        features = np.array([feature_row])
        prediction = model.predict(features)[0]
        
        # Apply reasonable constraints
        current_price = recent_data.iloc[-1]['Close']
        max_change = 0.1  # 10% max change per day
        min_price = current_price * (1 - max_change)
        max_price = current_price * (1 + max_change)
        
        prediction = np.clip(prediction, min_price, max_price)
        
        return prediction
    
    def save_model(self, model_name: str, symbol: str, additional_info: dict = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            symbol: Stock symbol the model was trained on
            additional_info: Additional metadata to save
            
        Returns:
            Path to the saved model file
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{symbol}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        model_data = {
            'model': self.trained_models[model_name],
            'model_name': model_name,
            'symbol': symbol,
            'timestamp': timestamp,
            'scores': self.model_scores.get(model_name, {}),
            'additional_info': additional_info or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model {model_name} for {symbol} saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> dict:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Dictionary containing model data
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model_name = model_data['model_name']
        self.trained_models[model_name] = model_data['model']
        self.model_scores[model_name] = model_data.get('scores', {})
        
        logger.info(f"Model {model_name} loaded from {filepath}")
        return model_data
    
    def list_saved_models(self) -> list:
        """List all saved model files."""
        if not os.path.exists(self.models_dir):
            return []
        
        model_files = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.models_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        model_data = pickle.load(f)
                    model_files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'model_name': model_data.get('model_name', 'unknown'),
                        'symbol': model_data.get('symbol', 'unknown'),
                        'timestamp': model_data.get('timestamp', 'unknown')
                    })
                except Exception as e:
                    logger.warning(f"Could not load model info from {filename}: {e}")
        
        return sorted(model_files, key=lambda x: x['timestamp'], reverse=True)
    
    def time_series_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
        """
        Split data using time series split (no random shuffling).
        """
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Time series split: {len(X_train)} training, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test

def demonstrate_prediction():
    """Demonstrate the improved prediction system."""
    from stockaroo.data.collector import StockDataCollector
    
    # Collect data
    collector = StockDataCollector("AAPL")
    data = collector.get_stock_data(period="1y", interval="1d")
    
    # Prepare data
    predictor = StockPredictor()
    X, y = predictor.prepare_time_series_data(data, lookback_window=10, prediction_horizon=1)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.time_series_split(X, y, test_size=0.2)
    
    # Train and evaluate
    predictor.train_model('linear_regression', X_train, y_train)
    results = predictor.evaluate_model('linear_regression', X_test, y_test)
    
    print(f"R² Score: {results['r2']:.4f}")
    print(f"RMSE: ${results['rmse']:.2f}")
    print(f"MAPE: {results['mape']:.2f}%")
    
    # Predict next day
    recent_data = data.tail(30)  # Use last 30 days
    next_day_pred = predictor.predict_next_day('linear_regression', recent_data)
    current_price = data.iloc[-1]['Close']
    
    print(f"\nCurrent price: ${current_price:.2f}")
    print(f"Predicted next day: ${next_day_pred:.2f}")
    print(f"Expected change: {((next_day_pred - current_price) / current_price * 100):+.2f}%")

if __name__ == "__main__":
    demonstrate_improved_prediction()
