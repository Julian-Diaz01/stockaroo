"""
Data collection module for stock data using Yahoo Finance API.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    A class to collect stock data from Yahoo Finance API.
    """
    
    def __init__(self, symbol: str = "AAPL"):
        """
        Initialize the data collector.
        
        Args:
            symbol (str): Stock symbol (default: AAPL for Apple)
        """
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        
    def get_stock_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            pd.DataFrame: Stock data with OHLCV information
        """
        try:
            logger.info(f"Fetching {self.symbol} data for period: {period}, interval: {interval}")
            
            # Fetch historical data
            data = self.ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Add some basic information
            data['Symbol'] = self.symbol
            data['Date'] = data.index
            
            # Calculate daily returns
            data['Daily_Return'] = data['Close'].pct_change()
            
            # Calculate price change
            data['Price_Change'] = data['Close'] - data['Open']
            
            # Calculate high-low spread
            data['HL_Spread'] = data['High'] - data['Low']
            
            # Calculate open-close spread
            data['OC_Spread'] = data['Close'] - data['Open']
            
            logger.info(f"Successfully fetched {len(data)} records for {self.symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {str(e)}")
            raise
    
    def get_company_info(self) -> dict:
        """
        Get company information.
        
        Returns:
            dict: Company information
        """
        try:
            info = self.ticker.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'current_price': info.get('currentPrice', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error fetching company info: {str(e)}")
            return {}
    
    def get_recent_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get recent stock data for the specified number of days.
        
        Args:
            days (int): Number of days to fetch
            
        Returns:
            pd.DataFrame: Recent stock data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_stock_data(
            period=f"{days}d",
            interval="1d"
        )

def main():
    """
    Example usage of the StockDataCollector.
    """
    # Create collector for Apple stock
    collector = StockDataCollector("AAPL")
    
    # Get 1 year of daily data
    data = collector.get_stock_data(period="1y", interval="1d")
    
    # Display basic information
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Get company info
    info = collector.get_company_info()
    print(f"\nCompany: {info.get('name', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")

if __name__ == "__main__":
    main()
