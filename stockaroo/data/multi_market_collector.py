"""
Multi-market data collector for cross-market predictions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MultiMarketCollector:
    """
    Collects data from multiple markets for cross-market analysis.
    """
    
    def __init__(self):
        """Initialize with market indices mapping."""
        self.market_indices = {
            'hong_kong': {
                'symbol': '^HSI',  # Hang Seng Index
                'name': 'Hong Kong Hang Seng',
                'timezone': 'Asia/Hong_Kong',
                'trading_hours': {'open': '09:30', 'close': '16:00'}
            },
            'europe': {
                'symbol': '^STOXX',  # STOXX Europe 600 - broader European market index
                'name': 'STOXX Europe 600',
                'timezone': 'Europe/Brussels',
                'trading_hours': {'open': '09:00', 'close': '17:30'}
            },
            'america': {
                'symbol': '^GSPC',  # S&P 500
                'name': 'US S&P 500',
                'timezone': 'America/New_York',
                'trading_hours': {'open': '09:30', 'close': '16:00'}
            }
        }
        
        # Alternative indices for more comprehensive coverage
        self.alternative_indices = {
            'hong_kong': ['^HSI', '^HSCE'],  # Hang Seng, Hang Seng China Enterprises
            'europe': ['^STOXX', '^FCHI', '^GDAXI', '^FTSE'],  # STOXX Europe 600, CAC 40, DAX, FTSE 100
            'america': ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
        }
    
    def get_market_data(self, market: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get data for a specific market.
        
        Args:
            market: Market name ('hong_kong', 'europe', 'america')
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with market data
        """
        if market not in self.market_indices:
            raise ValueError(f"Unknown market: {market}")
        
        symbol = self.market_indices[market]['symbol']
        logger.info(f"Fetching {market} data for symbol: {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Add market information
            data['Market'] = market
            data['Symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {market}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {market} data: {e}")
            raise
    
    def get_all_markets_data(self, period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get data for all markets.
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with market data
        """
        all_data = {}
        
        for market in self.market_indices.keys():
            try:
                all_data[market] = self.get_market_data(market, period, interval)
            except Exception as e:
                logger.warning(f"Failed to fetch {market} data: {e}")
                continue
        
        return all_data
    
    def align_market_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align market data by date for cross-market analysis.
        
        Args:
            market_data: Dictionary of market DataFrames
            
        Returns:
            Aligned DataFrame with all markets
        """
        if not market_data:
            raise ValueError("No market data provided")
        
        # Start with the first market's data
        aligned_data = None
        
        for market, data in market_data.items():
            if data.empty:
                continue
                
            # Select key columns and rename them with market prefix
            market_cols = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            market_cols.columns = [f"{market}_{col}" for col in market_cols.columns]
            
            if aligned_data is None:
                aligned_data = market_cols
            else:
                # Align by date (outer join to keep all dates)
                aligned_data = aligned_data.join(market_cols, how='outer')
        
        # Forward fill missing values (markets may have different trading days)
        aligned_data = aligned_data.ffill()
        
        # Drop rows where any market has missing data, but be more lenient
        # Only drop rows where more than 50% of values are missing
        missing_threshold = len(aligned_data.columns) * 0.5
        aligned_data = aligned_data.dropna(thresh=missing_threshold)
        
        logger.info(f"Aligned data shape: {aligned_data.shape}")
        return aligned_data
    
    def calculate_cross_market_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-market features for prediction.
        
        Args:
            aligned_data: Aligned market data
            
        Returns:
            DataFrame with cross-market features
        """
        features_data = aligned_data.copy()
        
        # Calculate price changes between markets
        for market in self.market_indices.keys():
            close_col = f"{market}_Close"
            if close_col in features_data.columns:
                # Daily returns
                features_data[f"{market}_returns"] = features_data[close_col].pct_change()
                
                # Volatility (rolling 5-day)
                features_data[f"{market}_volatility"] = features_data[f"{market}_returns"].rolling(5, min_periods=2).std()
        
        # Cross-market correlations (rolling 10-day) - only if we have enough data
        if len(features_data) >= 10:
            if 'hong_kong_returns' in features_data.columns and 'europe_returns' in features_data.columns:
                features_data['hk_eu_correlation'] = features_data['hong_kong_returns'].rolling(10, min_periods=5).corr(features_data['europe_returns'])
            
            if 'europe_returns' in features_data.columns and 'america_returns' in features_data.columns:
                features_data['eu_us_correlation'] = features_data['europe_returns'].rolling(10, min_periods=5).corr(features_data['america_returns'])
            
            if 'hong_kong_returns' in features_data.columns and 'america_returns' in features_data.columns:
                features_data['hk_us_correlation'] = features_data['hong_kong_returns'].rolling(10, min_periods=5).corr(features_data['america_returns'])
        else:
            # Add zero correlations if not enough data
            features_data['hk_eu_correlation'] = 0
            features_data['eu_us_correlation'] = 0
            features_data['hk_us_correlation'] = 0
        
        # Market lead-lag relationships
        for market in self.market_indices.keys():
            returns_col = f"{market}_returns"
            if returns_col in features_data.columns:
                # Previous day's return (lag 1)
                features_data[f"{market}_returns_lag1"] = features_data[returns_col].shift(1)
                # Previous 2 days' return (lag 2)
                features_data[f"{market}_returns_lag2"] = features_data[returns_col].shift(2)
        
        # Fill remaining NaN values with 0 (for rolling calculations that couldn't be computed)
        features_data = features_data.fillna(0)
        
        logger.info(f"Cross-market features shape: {features_data.shape}")
        return features_data
    
    def get_accumulation_distribution_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate accumulation/distribution indicators for each market.
        
        Args:
            market_data: Dictionary of market DataFrames
            
        Returns:
            DataFrame with A/D indicators
        """
        ad_data = {}
        
        for market, data in market_data.items():
            if data.empty:
                continue
            
            # Calculate Accumulation/Distribution Line
            # A/D = Previous A/D + ((Close - Low) - (High - Close)) / (High - Low) * Volume
            
            high_low_diff = data['High'] - data['Low']
            close_low_diff = data['Close'] - data['Low']
            high_close_diff = data['High'] - data['Close']
            
            # Avoid division by zero
            money_flow_multiplier = np.where(high_low_diff != 0, 
                                           (close_low_diff - high_close_diff) / high_low_diff, 
                                           0)
            
            money_flow_volume = money_flow_multiplier * data['Volume']
            ad_line = money_flow_volume.cumsum()
            
            # Calculate A/D Oscillator (10-day EMA - 3-day EMA)
            ad_ema_10 = ad_line.ewm(span=10).mean()
            ad_ema_3 = ad_line.ewm(span=3).mean()
            ad_oscillator = ad_ema_10 - ad_ema_3
            
            ad_data[f"{market}_ad_line"] = ad_line
            ad_data[f"{market}_ad_oscillator"] = ad_oscillator
            ad_data[f"{market}_ad_ema_10"] = ad_ema_10
            ad_data[f"{market}_ad_ema_3"] = ad_ema_3
        
        ad_df = pd.DataFrame(ad_data, index=data.index)
        return ad_df
    
    def prepare_cross_market_prediction_data(self, period: str = "1y", interval: str = "1d") -> tuple:
        """
        Prepare complete dataset for cross-market prediction.
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            Tuple of (features, target) for prediction
        """
        # Get all market data
        market_data = self.get_all_markets_data(period, interval)
        
        if len(market_data) < 2:
            raise ValueError("Need at least 2 markets for cross-market prediction")
        
        # Align data
        aligned_data = self.align_market_data(market_data)
        
        # Calculate cross-market features
        features_data = self.calculate_cross_market_features(aligned_data)
        
        # Skip A/D data for now to avoid NaN issues
        # ad_data = self.get_accumulation_distribution_data(market_data)
        
        # Use features data directly
        combined_data = features_data
        
        # Prepare features (exclude target columns)
        feature_cols = [col for col in combined_data.columns if not col.endswith('_Close')]
        features = combined_data[feature_cols].values
        
        # Target: Next day's US market close (if available)
        if 'america_Close' in combined_data.columns:
            target = combined_data['america_Close'].shift(-1)
            # Align features with target (remove last row since target is shifted)
            features = features[:-1]
            target = target[:-1]  # Remove last target value too
            # Remove any remaining NaN values
            valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
            features = features[valid_mask]
            target = target[valid_mask]
        else:
            raise ValueError("US market data required for target prediction")
        
        logger.info(f"Cross-market prediction data prepared: {features.shape[0]} samples, {features.shape[1]} features")
        
        if features.shape[0] == 0:
            logger.error("No valid samples found! Check data alignment and feature calculation.")
            logger.error(f"Combined data shape: {combined_data.shape}")
            logger.error(f"Combined data columns: {combined_data.columns.tolist()}")
            logger.error(f"Combined data has NaN: {combined_data.isnull().sum().sum()}")
        
        return features, target.values, combined_data
