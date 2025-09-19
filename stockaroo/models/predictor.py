"""
Improved Stock Predictor with proper time series handling.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
import pickle
import os
from datetime import datetime
import warnings

# Import gradient boosting models with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

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
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
        self.trained_models = {}
        self.model_scores = {}
        self.feature_names = None
        self.scaler = RobustScaler()  # Use RobustScaler for better handling of outliers
        self.models_dir = "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_time_series_data(self, data: pd.DataFrame, target_col: str = 'Close', 
                                lookback_window: int = 10, prediction_horizon: int = 1,
                                earnings_data: dict = None) -> tuple:
        """
        Prepare data for time series prediction with proper feature engineering.
        
        Args:
            data: Stock data
            target_col: Target column
            lookback_window: Number of past days to use as features
            prediction_horizon: Days ahead to predict
            earnings_data: Dictionary containing earnings data and impact analysis
            
        Returns:
            X, y: Features and targets
        """
        df = data.copy()
        
        # Create features using only past information
        features = []
        targets = []
        
        # Prepare earnings features if available
        earnings_features = self._prepare_earnings_features(df, earnings_data)
        
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
            
            # Enhanced technical indicators (calculated from past data only)
            if i >= 20:  # Ensure we have enough data for indicators
                # Simple moving averages
                ma_5 = df.iloc[i-5:i]['Close'].mean()
                ma_10 = df.iloc[i-10:i]['Close'].mean()
                ma_20 = df.iloc[i-20:i]['Close'].mean()
                
                # Exponential moving averages (more responsive to recent changes)
                ema_5 = df.iloc[i-5:i]['Close'].ewm(span=5).mean().iloc[-1]
                ema_10 = df.iloc[i-10:i]['Close'].ewm(span=10).mean().iloc[-1]
                
                # Price momentum and returns
                momentum_5 = (df.iloc[i-1]['Close'] / df.iloc[i-6]['Close'] - 1) if i >= 6 else 0
                momentum_10 = (df.iloc[i-1]['Close'] / df.iloc[i-11]['Close'] - 1) if i >= 11 else 0
                momentum_20 = (df.iloc[i-1]['Close'] / df.iloc[i-21]['Close'] - 1) if i >= 21 else 0
                
                # Volatility measures
                returns_5 = df.iloc[i-5:i]['Close'].pct_change().dropna()
                volatility_5 = returns_5.std() if len(returns_5) > 1 else 0
                
                returns_10 = df.iloc[i-10:i]['Close'].pct_change().dropna()
                volatility_10 = returns_10.std() if len(returns_10) > 1 else 0
                
                # Price position relative to moving averages
                price_vs_ma5 = (df.iloc[i-1]['Close'] / ma_5 - 1) if ma_5 > 0 else 0
                price_vs_ma10 = (df.iloc[i-1]['Close'] / ma_10 - 1) if ma_10 > 0 else 0
                price_vs_ma20 = (df.iloc[i-1]['Close'] / ma_20 - 1) if ma_20 > 0 else 0
                
                # Volume indicators
                volume_ma_5 = df.iloc[i-5:i]['Volume'].mean()
                volume_ratio = df.iloc[i-1]['Volume'] / volume_ma_5 if volume_ma_5 > 0 else 1
                
                # High-Low spread
                high_low_spread = (df.iloc[i-1]['High'] - df.iloc[i-1]['Low']) / df.iloc[i-1]['Close']
                
                # RSI-like momentum indicator
                gains = df.iloc[i-14:i]['Close'].diff().clip(lower=0).mean() if i >= 14 else 0
                losses = -df.iloc[i-14:i]['Close'].diff().clip(upper=0).mean() if i >= 14 else 0
                rs = gains / losses if losses > 0 else 0
                rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
                
                # Volatility (rolling std of returns)
                if i >= 10:
                    returns = df.iloc[i-10:i]['Close'].pct_change().dropna()
                    volatility = returns.std() if len(returns) > 1 else 0
                else:
                    volatility = 0
                
                feature_row.extend([
                    ma_5, ma_10, ma_20, ema_5, ema_10,
                    momentum_5, momentum_10, momentum_20,
                    volatility_5, volatility_10,
                    price_vs_ma5, price_vs_ma10, price_vs_ma20,
                    volume_ratio, high_low_spread, rsi
                ])
            else:
                # Pad with zeros if not enough data
                feature_row.extend([0] * 16)  # 16 new features
            
            # Add earnings features for this date
            current_date = df.index[i]
            earnings_feature_row = self._get_earnings_features_for_date(current_date, earnings_features)
            feature_row.extend(earnings_feature_row)
            
            features.append(feature_row)
            
            # Target: future price
            target = df.iloc[i + prediction_horizon - 1][target_col]
            targets.append(target)
        
        X = np.array(features)
        y = np.array(targets)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y
    
    def _prepare_earnings_features(self, df: pd.DataFrame, earnings_data: dict) -> dict:
        """
        Prepare earnings features for the dataset.
        
        Args:
            df: Stock price data
            earnings_data: Dictionary containing earnings data
            
        Returns:
            dict: Processed earnings features
        """
        earnings_features = {
            'impact_analysis': None,
            'upcoming_earnings': None,
            'earnings_surprise_history': []
        }
        
        if earnings_data is None:
            return earnings_features
        
        # Process earnings impact analysis if available
        if 'impact_analysis' in earnings_data and not earnings_data['impact_analysis'].empty:
            earnings_features['impact_analysis'] = earnings_data['impact_analysis']
        
        # Process upcoming earnings from calendar
        if 'calendar' in earnings_data and not earnings_data['calendar'].empty:
            earnings_features['upcoming_earnings'] = earnings_data['calendar']
        
        # Process earnings surprise history
        if 'history' in earnings_data and not earnings_data['history'].empty:
            history = earnings_data['history']
            for idx, row in history.iterrows():
                if 'Actual' in row and 'Estimate' in row:
                    if pd.notna(row['Actual']) and pd.notna(row['Estimate']) and row['Estimate'] != 0:
                        surprise = ((row['Actual'] - row['Estimate']) / abs(row['Estimate'])) * 100
                        earnings_features['earnings_surprise_history'].append({
                            'date': idx,
                            'surprise_pct': surprise,
                            'actual': row['Actual'],
                            'estimate': row['Estimate']
                        })
        
        return earnings_features
    
    def _get_earnings_features_for_date(self, current_date: pd.Timestamp, earnings_features: dict) -> list:
        """
        Get earnings features for a specific date.
        
        Args:
            current_date: Current date in the time series
            earnings_features: Processed earnings features
            
        Returns:
            list: Earnings features for the current date
        """
        feature_row = []
        
        # Days since last earnings announcement
        days_since_earnings = 0
        last_earnings_surprise = 0.0
        last_earnings_impact = 0.0
        
        if earnings_features['impact_analysis'] is not None:
            impact_df = earnings_features['impact_analysis']
            # Find the most recent earnings before current_date
            past_earnings = impact_df[impact_df['earnings_date'] < current_date]
            if not past_earnings.empty:
                last_earnings = past_earnings.iloc[-1]
                days_since_earnings = (current_date - last_earnings['earnings_date']).days
                last_earnings_surprise = last_earnings.get('earnings_surprise_pct', 0.0) or 0.0
                last_earnings_impact = last_earnings.get('price_change_pct', 0.0) or 0.0
        
        # Days until next earnings announcement
        days_until_earnings = 365  # Default to 1 year if no upcoming earnings
        if earnings_features['upcoming_earnings'] is not None:
            upcoming = earnings_features['upcoming_earnings']
            future_earnings = upcoming[upcoming.index > current_date]
            if not future_earnings.empty:
                next_earnings = future_earnings.iloc[0]
                days_until_earnings = (next_earnings.name - current_date).days
        
        # Average earnings surprise over last 4 quarters
        avg_earnings_surprise = 0.0
        if earnings_features['earnings_surprise_history']:
            recent_surprises = [
                s['surprise_pct'] for s in earnings_features['earnings_surprise_history']
                if s['date'] < current_date
            ][-4:]  # Last 4 quarters
            if recent_surprises:
                avg_earnings_surprise = np.mean(recent_surprises)
        
        # Earnings volatility (std of recent surprises)
        earnings_volatility = 0.0
        if earnings_features['earnings_surprise_history']:
            recent_surprises = [
                s['surprise_pct'] for s in earnings_features['earnings_surprise_history']
                if s['date'] < current_date
            ][-8:]  # Last 8 quarters
            if len(recent_surprises) > 1:
                earnings_volatility = np.std(recent_surprises)
        
        # Apply scaling and normalization for earnings features to account for rarity
        # Earnings happen only 4 times per year, so we need to scale their impact appropriately
        
        # Normalize days since/until earnings (0-1 scale)
        days_since_earnings_norm = min(days_since_earnings / 90, 1.0)  # Cap at 90 days
        days_until_earnings_norm = min(days_until_earnings / 90, 1.0)  # Cap at 90 days
        
        # Scale earnings surprise and impact by their historical volatility
        # This prevents rare large surprises from dominating the model
        surprise_scale_factor = 0.1  # Reduce impact of earnings surprises
        impact_scale_factor = 0.1    # Reduce impact of price changes
        
        last_earnings_surprise_scaled = last_earnings_surprise * surprise_scale_factor
        last_earnings_impact_scaled = last_earnings_impact * impact_scale_factor
        avg_earnings_surprise_scaled = avg_earnings_surprise * surprise_scale_factor
        earnings_volatility_scaled = earnings_volatility * surprise_scale_factor
        
        # Add binary indicators for earnings proximity (more robust than continuous features)
        near_earnings = 1 if days_until_earnings <= 30 else 0  # Within 30 days of earnings
        recent_earnings = 1 if days_since_earnings <= 7 else 0  # Within 7 days of earnings
        
        feature_row.extend([
            days_since_earnings_norm,
            days_until_earnings_norm,
            last_earnings_surprise_scaled,
            last_earnings_impact_scaled,
            avg_earnings_surprise_scaled,
            earnings_volatility_scaled,
            near_earnings,
            recent_earnings
        ])
        
        return feature_row
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Train a specific model with feature scaling."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        logger.info(f"Training {model_name} model")
        
        # Scale features for models that benefit from it
        models_need_scaling = ['linear_regression', 'ridge', 'lasso', 'xgboost', 'lightgbm']
        if model_name in models_need_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        # Get the model from the models dictionary
        model = self.models[model_name]
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        # Calculate training score
        train_score = model.score(X_train_scaled, y_train)
        
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
        
        # Scale test features if the model was trained with scaling
        models_need_scaling = ['linear_regression', 'ridge', 'lasso', 'xgboost', 'lightgbm']
        if model_name in models_need_scaling:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Robust MAPE calculation to handle edge cases
        # Avoid division by zero and handle very small values
        y_test_abs = np.abs(y_test)
        # Use a small threshold to avoid division by very small numbers
        threshold = np.percentile(y_test_abs, 5)  # 5th percentile as threshold
        valid_indices = y_test_abs > threshold
        
        if np.sum(valid_indices) > 0:
            # Calculate MAPE only for valid indices
            mape = np.mean(np.abs((y_test[valid_indices] - y_pred[valid_indices]) / y_test[valid_indices])) * 100
        else:
            # Fallback: use symmetric MAPE (sMAPE) which is more robust
            numerator = np.abs(y_test - y_pred)
            denominator = (np.abs(y_test) + np.abs(y_pred)) / 2
            # Avoid division by zero
            denominator = np.where(denominator == 0, 1e-8, denominator)
            mape = np.mean(numerator / denominator) * 100
        
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
                        lookback_window: int = 10, earnings_data: dict = None) -> float:
        """
        Predict the next day's price using the most recent data.
        
        Args:
            model_name: Name of the trained model
            recent_data: Recent stock data (last lookback_window + 20 days)
            lookback_window: Number of past days to use as features
            earnings_data: Dictionary containing earnings data
            
        Returns:
            Predicted price for the next day
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Prepare earnings features if available
        earnings_features = self._prepare_earnings_features(recent_data, earnings_data)
        
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
        
        # Enhanced technical indicators (same as in prepare_time_series_data)
        if len(recent_data) >= 20:
            # Simple moving averages
            ma_5 = recent_data.iloc[-5:]['Close'].mean()
            ma_10 = recent_data.iloc[-10:]['Close'].mean()
            ma_20 = recent_data.iloc[-20:]['Close'].mean()
            
            # Exponential moving averages
            ema_5 = recent_data.iloc[-5:]['Close'].ewm(span=5).mean().iloc[-1]
            ema_10 = recent_data.iloc[-10:]['Close'].ewm(span=10).mean().iloc[-1]
            
            # Price momentum and returns
            momentum_5 = (recent_data.iloc[-1]['Close'] / recent_data.iloc[-6]['Close'] - 1) if len(recent_data) >= 6 else 0
            momentum_10 = (recent_data.iloc[-1]['Close'] / recent_data.iloc[-11]['Close'] - 1) if len(recent_data) >= 11 else 0
            momentum_20 = (recent_data.iloc[-1]['Close'] / recent_data.iloc[-21]['Close'] - 1) if len(recent_data) >= 21 else 0
            
            # Volatility measures
            returns_5 = recent_data.iloc[-5:]['Close'].pct_change().dropna()
            volatility_5 = returns_5.std() if len(returns_5) > 1 else 0
            
            returns_10 = recent_data.iloc[-10:]['Close'].pct_change().dropna()
            volatility_10 = returns_10.std() if len(returns_10) > 1 else 0
            
            # Price position relative to moving averages
            price_vs_ma5 = (recent_data.iloc[-1]['Close'] / ma_5 - 1) if ma_5 > 0 else 0
            price_vs_ma10 = (recent_data.iloc[-1]['Close'] / ma_10 - 1) if ma_10 > 0 else 0
            price_vs_ma20 = (recent_data.iloc[-1]['Close'] / ma_20 - 1) if ma_20 > 0 else 0
            
            # Volume indicators
            volume_ma_5 = recent_data.iloc[-5:]['Volume'].mean()
            volume_ratio = recent_data.iloc[-1]['Volume'] / volume_ma_5 if volume_ma_5 > 0 else 1
            
            # High-Low spread
            high_low_spread = (recent_data.iloc[-1]['High'] - recent_data.iloc[-1]['Low']) / recent_data.iloc[-1]['Close']
            
            # RSI-like momentum indicator
            if len(recent_data) >= 14:
                gains = recent_data.iloc[-14:]['Close'].diff().clip(lower=0).mean()
                losses = -recent_data.iloc[-14:]['Close'].diff().clip(upper=0).mean()
                rs = gains / losses if losses > 0 else 0
                rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
            else:
                rsi = 50
                
            feature_row.extend([
                ma_5, ma_10, ma_20, ema_5, ema_10,
                momentum_5, momentum_10, momentum_20,
                volatility_5, volatility_10,
                price_vs_ma5, price_vs_ma10, price_vs_ma20,
                volume_ratio, high_low_spread, rsi
            ])
        else:
            feature_row.extend([0] * 16)  # 16 new features
        
        # Add earnings features for current date
        current_date = recent_data.index[-1]
        earnings_feature_row = self._get_earnings_features_for_date(current_date, earnings_features)
        feature_row.extend(earnings_feature_row)
        
        # Make prediction
        features = np.array([feature_row])
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features if the model was trained with scaling
        models_need_scaling = ['linear_regression', 'ridge', 'lasso', 'xgboost', 'lightgbm']
        if model_name in models_need_scaling:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
            
        prediction = model.predict(features_scaled)[0]
        
        # Apply volatility-based constraints instead of arbitrary clipping
        current_price = recent_data.iloc[-1]['Close']
        
        # Calculate historical volatility (20-day rolling standard deviation of returns)
        if len(recent_data) >= 20:
            returns = recent_data['Close'].pct_change().dropna()
            volatility = returns.tail(20).std()
            
            # Use 2-sigma bounds (covers ~95% of normal price movements)
            # Add some buffer for extreme market conditions
            max_change = min(3 * volatility, 0.15)  # Cap at 15% even in extreme volatility
            min_change = -max_change  # Symmetric bounds
        else:
            # Fallback to conservative bounds if insufficient data
            max_change = 0.1
            min_change = -0.1
        
        min_price = current_price * (1 + min_change)
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
    
    def time_series_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                         embargo_period: int = 0) -> tuple:
        """
        Split data using time series split with embargo period to prevent data leakage.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing
            embargo_period: Number of samples to skip between train and test sets
            
        Returns:
            X_train, X_test, y_train, y_test: Split data with embargo period
        """
        total_samples = len(X)
        test_samples = int(total_samples * test_size)
        train_samples = total_samples - test_samples - embargo_period
        
        # Ensure we have enough data after applying embargo
        if train_samples <= 0 or test_samples <= 0:
            raise ValueError(f"Not enough data for split with embargo. "
                           f"Total: {total_samples}, Train: {train_samples}, "
                           f"Test: {test_samples}, Embargo: {embargo_period}")
        
        # Split with embargo period
        X_train = X[:train_samples]
        X_test = X[train_samples + embargo_period:]
        y_train = y[:train_samples]
        y_test = y[train_samples + embargo_period:]
        
        logger.info(f"Time series split with embargo: {len(X_train)} training, "
                   f"{len(X_test)} test samples, {embargo_period} embargo samples")
        
        return X_train, X_test, y_train, y_test
    
    def time_series_split_with_validation(self, X: np.ndarray, y: np.ndarray, 
                                        train_size: float = 0.6, val_size: float = 0.2,
                                        embargo_period: int = 5) -> tuple:
        """
        Split data into train/validation/test sets with embargo periods to prevent data leakage.
        
        Args:
            X: Feature matrix
            y: Target vector
            train_size: Fraction of data to use for training
            val_size: Fraction of data to use for validation
            embargo_period: Number of samples to skip between each set
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: Split data with embargo periods
        """
        total_samples = len(X)
        train_samples = int(total_samples * train_size)
        val_samples = int(total_samples * val_size)
        test_samples = total_samples - train_samples - val_samples - (2 * embargo_period)
        
        # Ensure we have enough data after applying embargo periods
        if train_samples <= 0 or val_samples <= 0 or test_samples <= 0:
            raise ValueError(f"Not enough data for split with embargo. "
                           f"Total: {total_samples}, Train: {train_samples}, "
                           f"Val: {val_samples}, Test: {test_samples}, "
                           f"Embargo: {embargo_period}")
        
        # Calculate split indices with embargo periods
        train_end = train_samples
        val_start = train_end + embargo_period
        val_end = val_start + val_samples
        test_start = val_end + embargo_period
        
        # Split data
        X_train = X[:train_end]
        X_val = X[val_start:val_end]
        X_test = X[test_start:]
        y_train = y[:train_end]
        y_val = y[val_start:val_end]
        y_test = y[test_start:]
        
        logger.info(f"Time series split with validation and embargo:")
        logger.info(f"  Train: {len(X_train)} samples (0 to {train_end-1})")
        logger.info(f"  Embargo: {embargo_period} samples ({train_end} to {val_start-1})")
        logger.info(f"  Validation: {len(X_val)} samples ({val_start} to {val_end-1})")
        logger.info(f"  Embargo: {embargo_period} samples ({val_end} to {test_start-1})")
        logger.info(f"  Test: {len(X_test)} samples ({test_start} to {total_samples-1})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def validate_no_data_leakage(self, X_train: np.ndarray, X_test: np.ndarray, 
                                y_train: np.ndarray, y_test: np.ndarray) -> bool:
        """
        Validate that there's no data leakage between train and test sets.
        
        Args:
            X_train, X_test: Training and test feature matrices
            y_train, y_test: Training and test target vectors
            
        Returns:
            True if no leakage detected, False otherwise
        """
        # Check for overlapping samples by comparing feature vectors
        leakage_detected = False
        
        for i, train_sample in enumerate(X_train):
            for j, test_sample in enumerate(X_test):
                if np.array_equal(train_sample, test_sample):
                    logger.warning(f"Data leakage detected: Train sample {i} matches test sample {j}")
                    leakage_detected = True
        
        # Check for overlapping targets
        for i, train_target in enumerate(y_train):
            for j, test_target in enumerate(y_test):
                if np.array_equal(train_target, test_target):
                    logger.warning(f"Target leakage detected: Train target {i} matches test target {j}")
                    leakage_detected = True
        
        if not leakage_detected:
            logger.info("No data leakage detected between train and test sets")
        
        return not leakage_detected
    
    def get_split_info(self, X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Get information about the data split for debugging and validation.
        
        Returns:
            Dictionary with split information
        """
        return {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X_train) + len(X_test),
            'train_ratio': len(X_train) / (len(X_train) + len(X_test)),
            'test_ratio': len(X_test) / (len(X_train) + len(X_test)),
            'feature_dimensions': X_train.shape[1] if len(X_train) > 0 else 0,
            'no_leakage': self.validate_no_data_leakage(X_train, X_test, y_train, y_test)
        }

def demonstrate_prediction():
    """Demonstrate the improved prediction system with data leakage prevention and earnings integration."""
    from stockaroo.data.collector import StockDataCollector
    
    # Collect data
    collector = StockDataCollector("AAPL")
    data = collector.get_stock_data(period="1y", interval="1d")
    
    # Collect earnings data
    earnings_data = collector.get_earnings_data()
    earnings_impact = collector.get_earnings_impact_analysis(data, earnings_data)
    
    # Combine earnings data
    combined_earnings_data = {
        **earnings_data,
        'impact_analysis': earnings_impact
    }
    
    # Prepare data with earnings features
    predictor = StockPredictor()
    X, y = predictor.prepare_time_series_data(
        data, 
        lookback_window=10, 
        prediction_horizon=1,
        earnings_data=combined_earnings_data
    )
    
    print(f"Total samples prepared: {len(X)}")
    
    # Split data with embargo period to prevent data leakage
    embargo_period = 5  # Skip 5 samples between train and test
    X_train, X_test, y_train, y_test = predictor.time_series_split(
        X, y, test_size=0.2, embargo_period=embargo_period
    )
    
    # Validate no data leakage
    split_info = predictor.get_split_info(X_train, X_test, y_train, y_test)
    print(f"\nSplit Information:")
    print(f"  Train samples: {split_info['train_samples']}")
    print(f"  Test samples: {split_info['test_samples']}")
    print(f"  Train ratio: {split_info['train_ratio']:.2%}")
    print(f"  Test ratio: {split_info['test_ratio']:.2%}")
    print(f"  No data leakage: {split_info['no_leakage']}")
    
    # Train and evaluate
    predictor.train_model('linear_regression', X_train, y_train)
    results = predictor.evaluate_model('linear_regression', X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {results['r2']:.4f}")
    print(f"  RMSE: ${results['rmse']:.2f}")
    print(f"  MAPE: {results['mape']:.2f}%")
    
    # Demonstrate train/validation/test split with embargo
    print(f"\n--- Demonstrating Train/Validation/Test Split with Embargo ---")
    X_train_val, X_val, X_test_val, y_train_val, y_val, y_test_val = predictor.time_series_split_with_validation(
        X, y, train_size=0.6, val_size=0.2, embargo_period=3
    )
    
    # Train on training set
    predictor.train_model('ridge', X_train_val, y_train_val)
    
    # Evaluate on validation set
    val_results = predictor.evaluate_model('ridge', X_val, y_val)
    print(f"Validation Performance:")
    print(f"  R² Score: {val_results['r2']:.4f}")
    print(f"  RMSE: ${val_results['rmse']:.2f}")
    
    # Final evaluation on test set
    test_results = predictor.evaluate_model('ridge', X_test_val, y_test_val)
    print(f"Test Performance:")
    print(f"  R² Score: {test_results['r2']:.4f}")
    print(f"  RMSE: ${test_results['rmse']:.2f}")
    
    # Predict next day with earnings data
    recent_data = data.tail(30)  # Use last 30 days
    next_day_pred = predictor.predict_next_day(
        'linear_regression', 
        recent_data, 
        lookback_window=10,
        earnings_data=combined_earnings_data
    )
    current_price = data.iloc[-1]['Close']
    
    print(f"\nPrediction:")
    print(f"  Current price: ${current_price:.2f}")
    print(f"  Predicted next day: ${next_day_pred:.2f}")
    print(f"  Expected change: {((next_day_pred - current_price) / current_price * 100):+.2f}%")
    
    # Display earnings information if available
    if earnings_impact is not None and not earnings_impact.empty:
        print(f"\nEarnings Analysis:")
        print(f"  Total earnings events analyzed: {len(earnings_impact)}")
        if len(earnings_impact) > 0:
            avg_surprise = earnings_impact['earnings_surprise_pct'].mean()
            avg_impact = earnings_impact['price_change_pct'].mean()
            print(f"  Average earnings surprise: {avg_surprise:.2f}%")
            print(f"  Average price impact: {avg_impact:.2f}%")
    else:
        print(f"\nEarnings Analysis: No earnings data available for analysis")

def rolling_backtest(self, data: pd.DataFrame, target_col: str = 'Close', 
                    lookback_window: int = 10, prediction_horizon: int = 1,
                    earnings_data: dict = None, train_size: int = 100, 
                    test_size: int = 20, step_size: int = 5) -> dict:
        """
        Perform rolling backtest (walk-forward validation) to evaluate model performance
        across multiple time periods.
        
        Args:
            data: Stock data
            target_col: Target column
            lookback_window: Number of past days to use as features
            prediction_horizon: Days ahead to predict
            earnings_data: Dictionary containing earnings data
            train_size: Number of days for training in each fold
            test_size: Number of days for testing in each fold
            step_size: Number of days to move forward in each iteration
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting rolling backtest: train_size={train_size}, test_size={test_size}, step_size={step_size}")
        
        # Prepare data with returns instead of absolute prices for stationarity
        df = data.copy()
        
        # Calculate returns (percentage change) for better stationarity
        df['Returns'] = df[target_col].pct_change()
        df['Log_Returns'] = np.log(df[target_col] / df[target_col].shift(1))
        
        # Use returns as target instead of absolute prices
        target_col_returns = 'Returns'
        
        # Prepare earnings features
        earnings_features = self._prepare_earnings_features(df, earnings_data)
        
        # Initialize results storage
        backtest_results = {
            'fold_results': [],
            'model_performance': {},
            'predictions': [],
            'actuals': [],
            'dates': [],
            'fold_info': []
        }
        
        # Calculate number of folds
        total_days = len(df)
        min_required = lookback_window + train_size + test_size + prediction_horizon
        
        if total_days < min_required:
            logger.warning(f"Insufficient data for rolling backtest. Need {min_required}, have {total_days}")
            return backtest_results
        
        # Perform rolling backtest
        start_idx = lookback_window + train_size
        end_idx = total_days - test_size - prediction_horizon
        
        fold_count = 0
        for start_test in range(start_idx, end_idx, step_size):
            end_test = min(start_test + test_size, total_days - prediction_horizon)
            
            if end_test - start_test < test_size:
                break
                
            fold_count += 1
            logger.info(f"Processing fold {fold_count}: test period {start_test}-{end_test}")
            
            # Define training and test periods
            train_start = start_test - train_size
            train_end = start_test
            
            # Prepare training data
            train_data = df.iloc[train_start:train_end].copy()
            test_data = df.iloc[start_test:end_test].copy()
            
            # Create features and targets for training
            X_train, y_train = self._prepare_features_targets(
                train_data, target_col_returns, lookback_window, prediction_horizon, earnings_features
            )
            
            # Create features and targets for testing
            X_test, y_test = self._prepare_features_targets(
                test_data, target_col_returns, lookback_window, prediction_horizon, earnings_features
            )
            
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"Skipping fold {fold_count}: insufficient data")
                continue
            
            # Train and evaluate each model
            fold_results = {}
            
            for model_name in self.models.keys():
                try:
                    # Train model
                    self.train_model(model_name, X_train, y_train)
                    
                    # Evaluate model
                    eval_results = self.evaluate_model(model_name, X_test, y_test)
                    
                    # Store results
                    fold_results[model_name] = eval_results
                    
                    # Store predictions and actuals for analysis
                    if model_name not in backtest_results['model_performance']:
                        backtest_results['model_performance'][model_name] = {
                            'predictions': [],
                            'actuals': [],
                            'dates': [],
                            'metrics': []
                        }
                    
                    backtest_results['model_performance'][model_name]['predictions'].extend(eval_results['predictions'])
                    backtest_results['model_performance'][model_name]['actuals'].extend(y_test)
                    backtest_results['model_performance'][model_name]['dates'].extend(test_data.index[lookback_window:])
                    backtest_results['model_performance'][model_name]['metrics'].append({
                        'fold': fold_count,
                        'r2': eval_results['r2'],
                        'mae': eval_results['mae'],
                        'rmse': eval_results['rmse'],
                        'mape': eval_results['mape']
                    })
                    
                except Exception as e:
                    logger.warning(f"Error in fold {fold_count} for model {model_name}: {e}")
                    continue
            
            # Store fold information
            backtest_results['fold_results'].append(fold_results)
            backtest_results['fold_info'].append({
                'fold': fold_count,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': start_test,
                'test_end': end_test,
                'train_dates': (train_data.index[0], train_data.index[-1]),
                'test_dates': (test_data.index[0], test_data.index[-1])
            })
        
        logger.info(f"Rolling backtest completed: {fold_count} folds processed")
        return backtest_results
    
def _prepare_features_targets(self, data: pd.DataFrame, target_col: str, 
                                 lookback_window: int, prediction_horizon: int, 
                                 earnings_features: dict) -> tuple:
        """Helper method to prepare features and targets for a given dataset."""
        
        features = []
        targets = []
        
        for i in range(lookback_window, len(data) - prediction_horizon + 1):
            # Features: past lookback_window days
            feature_row = []
            
            # Price features (past days) - using returns for stationarity
            for j in range(lookback_window):
                idx = i - lookback_window + j
                if idx >= 0:
                    feature_row.extend([
                        data.iloc[idx]['Open'],
                        data.iloc[idx]['High'], 
                        data.iloc[idx]['Low'],
                        data.iloc[idx]['Close'],
                        data.iloc[idx]['Volume']
                    ])
                else:
                    feature_row.extend([0, 0, 0, 0, 0])
            
            # Enhanced technical indicators
            if i >= 20:
                # Calculate technical indicators (same as before)
                ma_5 = data.iloc[i-5:i]['Close'].mean()
                ma_10 = data.iloc[i-10:i]['Close'].mean()
                ma_20 = data.iloc[i-20:i]['Close'].mean()
                
                # Exponential moving averages
                ema_5 = data.iloc[i-5:i]['Close'].ewm(span=5).mean().iloc[-1]
                ema_10 = data.iloc[i-10:i]['Close'].ewm(span=10).mean().iloc[-1]
                
                # Price momentum and returns
                momentum_5 = (data.iloc[i-1]['Close'] / data.iloc[i-6]['Close'] - 1) if i >= 6 else 0
                momentum_10 = (data.iloc[i-1]['Close'] / data.iloc[i-11]['Close'] - 1) if i >= 11 else 0
                momentum_20 = (data.iloc[i-1]['Close'] / data.iloc[i-21]['Close'] - 1) if i >= 21 else 0
                
                # Volatility measures
                returns_5 = data.iloc[i-5:i]['Close'].pct_change().dropna()
                volatility_5 = returns_5.std() if len(returns_5) > 1 else 0
                
                returns_10 = data.iloc[i-10:i]['Close'].pct_change().dropna()
                volatility_10 = returns_10.std() if len(returns_10) > 1 else 0
                
                # Price position relative to moving averages
                price_vs_ma5 = (data.iloc[i-1]['Close'] / ma_5 - 1) if ma_5 > 0 else 0
                price_vs_ma10 = (data.iloc[i-1]['Close'] / ma_10 - 1) if ma_10 > 0 else 0
                price_vs_ma20 = (data.iloc[i-1]['Close'] / ma_20 - 1) if ma_20 > 0 else 0
                
                # Volume indicators
                volume_ma_5 = data.iloc[i-5:i]['Volume'].mean()
                volume_ratio = data.iloc[i-1]['Volume'] / volume_ma_5 if volume_ma_5 > 0 else 1
                
                # High-Low spread
                high_low_spread = (data.iloc[i-1]['High'] - data.iloc[i-1]['Low']) / data.iloc[i-1]['Close']
                
                # RSI-like momentum indicator
                if i >= 14:
                    gains = data.iloc[i-14:i]['Close'].diff().clip(lower=0).mean()
                    losses = -data.iloc[i-14:i]['Close'].diff().clip(upper=0).mean()
                    rs = gains / losses if losses > 0 else 0
                    rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
                else:
                    rsi = 50
                    
                feature_row.extend([
                    ma_5, ma_10, ma_20, ema_5, ema_10,
                    momentum_5, momentum_10, momentum_20,
                    volatility_5, volatility_10,
                    price_vs_ma5, price_vs_ma10, price_vs_ma20,
                    volume_ratio, high_low_spread, rsi
                ])
            else:
                feature_row.extend([0] * 16)
            
            # Add earnings features for this date
            current_date = data.index[i]
            earnings_feature_row = self._get_earnings_features_for_date(current_date, earnings_features)
            feature_row.extend(earnings_feature_row)
            
            features.append(feature_row)
            
            # Target: future returns instead of absolute price
            target = data.iloc[i + prediction_horizon - 1][target_col]
            targets.append(target)
        
        X = np.array(features)
        y = np.array(targets)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y

if __name__ == "__main__":
    demonstrate_prediction()
