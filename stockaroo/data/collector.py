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
    
    def get_earnings_data(self) -> dict:
        """
        Get earnings data including calendar, estimates, and actual results.
        
        Returns:
            dict: Earnings data with calendar, estimates, and actual results
        """
        try:
            earnings_data = {}
            
            # Get earnings calendar
            try:
                earnings_calendar = self.ticker.calendar
                if earnings_calendar is not None and not earnings_calendar.empty:
                    earnings_data['calendar'] = earnings_calendar
                    logger.info(f"Retrieved earnings calendar for {self.symbol}")
                else:
                    logger.warning(f"No earnings calendar data available for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch earnings calendar: {e}")
            
            # Get earnings history
            try:
                earnings_history = self.ticker.earnings_history
                if earnings_history is not None and not earnings_history.empty:
                    earnings_data['history'] = earnings_history
                    logger.info(f"Retrieved earnings history for {self.symbol}")
                else:
                    logger.warning(f"No earnings history data available for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch earnings history: {e}")
            
            # Get quarterly earnings
            try:
                quarterly_earnings = self.ticker.quarterly_earnings
                if quarterly_earnings is not None and not quarterly_earnings.empty:
                    earnings_data['quarterly'] = quarterly_earnings
                    logger.info(f"Retrieved quarterly earnings for {self.symbol}")
                else:
                    logger.warning(f"No quarterly earnings data available for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch quarterly earnings: {e}")
            
            # Get financials for additional earnings metrics
            try:
                financials = self.ticker.financials
                if financials is not None and not financials.empty:
                    earnings_data['financials'] = financials
                    logger.info(f"Retrieved financials for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch financials: {e}")
            
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error fetching earnings data for {self.symbol}: {str(e)}")
            return {}
    
    def get_earnings_impact_analysis(self, stock_data: pd.DataFrame, earnings_data: dict) -> pd.DataFrame:
        """
        Analyze the impact of earnings announcements on stock prices.
        
        Args:
            stock_data: Historical stock price data
            earnings_data: Earnings data dictionary
            
        Returns:
            pd.DataFrame: Analysis of earnings impact on stock prices
        """
        try:
            impact_analysis = []
            
            if 'history' in earnings_data and not earnings_data['history'].empty:
                earnings_history = earnings_data['history']
                
                for idx, row in earnings_history.iterrows():
                    earnings_date = idx
                    
                    # Find stock price data around earnings date
                    # Look for price data within 5 days of earnings announcement
                    date_range = pd.date_range(
                        start=earnings_date - timedelta(days=5),
                        end=earnings_date + timedelta(days=5),
                        freq='D'
                    )
                    
                    # Get stock prices around earnings date
                    pre_earnings_price = None
                    post_earnings_price = None
                    
                    for check_date in date_range:
                        if check_date in stock_data.index:
                            if check_date < earnings_date and pre_earnings_price is None:
                                pre_earnings_price = stock_data.loc[check_date, 'Close']
                            elif check_date > earnings_date and post_earnings_price is None:
                                post_earnings_price = stock_data.loc[check_date, 'Close']
                    
                    if pre_earnings_price and post_earnings_price:
                        price_change = post_earnings_price - pre_earnings_price
                        price_change_pct = (price_change / pre_earnings_price) * 100
                        
                        # Calculate earnings surprise if estimates are available
                        earnings_surprise = None
                        if 'Actual' in row and 'Estimate' in row:
                            if pd.notna(row['Actual']) and pd.notna(row['Estimate']) and row['Estimate'] != 0:
                                earnings_surprise = ((row['Actual'] - row['Estimate']) / abs(row['Estimate'])) * 100
                        
                        impact_analysis.append({
                            'earnings_date': earnings_date,
                            'actual_eps': row.get('Actual', None),
                            'estimated_eps': row.get('Estimate', None),
                            'earnings_surprise_pct': earnings_surprise,
                            'pre_earnings_price': pre_earnings_price,
                            'post_earnings_price': post_earnings_price,
                            'price_change': price_change,
                            'price_change_pct': price_change_pct,
                            'volume': row.get('Volume', None)
                        })
            
            if impact_analysis:
                impact_df = pd.DataFrame(impact_analysis)
                impact_df = impact_df.sort_values('earnings_date')
                logger.info(f"Analyzed {len(impact_df)} earnings events for {self.symbol}")
                return impact_df
            else:
                logger.warning(f"No earnings impact analysis possible for {self.symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error analyzing earnings impact for {self.symbol}: {str(e)}")
            return pd.DataFrame()

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
