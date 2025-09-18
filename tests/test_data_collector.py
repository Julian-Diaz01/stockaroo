"""
Tests for data collection module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from stockaroo.data import StockDataCollector


class TestStockDataCollector:
    """Test cases for StockDataCollector."""
    
    def test_init(self):
        """Test collector initialization."""
        collector = StockDataCollector("AAPL")
        assert collector.symbol == "AAPL"
        assert collector.ticker is not None
    
    @patch('stockaroo.data.collector.yf.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """Test successful data collection."""
        # Mock data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector("AAPL")
        result = collector.get_stock_data(period="1y", interval="1d")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'Daily_Return' in result.columns
        assert 'Price_Change' in result.columns
    
    @patch('stockaroo.data.collector.yf.Ticker')
    def test_get_stock_data_empty(self, mock_ticker):
        """Test handling of empty data."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector("INVALID")
        
        with pytest.raises(ValueError, match="No data found"):
            collector.get_stock_data(period="1y", interval="1d")
    
    @patch('stockaroo.data.collector.yf.Ticker')
    def test_get_company_info(self, mock_ticker):
        """Test company info retrieval."""
        mock_info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'currentPrice': 150.0
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector("AAPL")
        result = collector.get_company_info()
        
        assert result['name'] == 'Apple Inc.'
        assert result['sector'] == 'Technology'
        assert result['industry'] == 'Consumer Electronics'
