"""
Visualization module for stock data analysis and predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockVisualizer:
    """
    A class for creating visualizations of stock data and predictions.
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Default figure size
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_price_data(self, data: pd.DataFrame, symbol: str = "AAPL", 
                       save_path: str = None) -> None:
        """
        Plot stock price data with volume.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            save_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2, color=self.colors[0])
        ax1.plot(data.index, data['Open'], label='Open Price', alpha=0.7, color=self.colors[1])
        ax1.fill_between(data.index, data['Low'], data['High'], alpha=0.3, color=self.colors[2])
        
        ax1.set_title(f'{symbol} Stock Price Over Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        ax2.bar(data.index, data['Volume'], alpha=0.7, color=self.colors[3])
        ax2.set_title('Trading Volume', fontsize=14)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Price plot saved to {save_path}")
        
        plt.show()
    
    def plot_technical_indicators(self, data: pd.DataFrame, symbol: str = "AAPL",
                                save_path: str = None) -> None:
        """
        Plot technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            symbol (str): Stock symbol
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Price with moving averages
        axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
        if 'MA_20' in data.columns:
            axes[0].plot(data.index, data['MA_20'], label='MA 20', alpha=0.8)
        if 'MA_50' in data.columns:
            axes[0].plot(data.index, data['MA_50'], label='MA 50', alpha=0.8)
        axes[0].set_title(f'{symbol} Price with Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in data.columns:
            axes[1].plot(data.index, data['MACD'], label='MACD', linewidth=2)
            axes[1].plot(data.index, data['MACD_Signal'], label='Signal', linewidth=2)
            axes[1].bar(data.index, data['MACD_Histogram'], label='Histogram', alpha=0.6)
            axes[1].set_title('MACD Indicator')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in data.columns:
            axes[2].plot(data.index, data['RSI'], label='RSI', linewidth=2, color='purple')
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            axes[2].set_title('RSI Indicator')
            axes[2].set_ylabel('RSI')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns:
            axes[3].plot(data.index, data['Close'], label='Close Price', linewidth=2)
            axes[3].plot(data.index, data['BB_Upper'], label='Upper Band', alpha=0.7)
            axes[3].plot(data.index, data['BB_Lower'], label='Lower Band', alpha=0.7)
            axes[3].fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.2)
            axes[3].set_title('Bollinger Bands')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Technical indicators plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_test: np.ndarray, predictions: dict, 
                        model_names: list = None, save_path: str = None) -> None:
        """
        Plot model predictions vs actual values.
        
        Args:
            y_test (np.ndarray): Actual test values
            predictions (dict): Dictionary of model predictions
            model_names (list): List of model names to plot
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(predictions.keys())
        
        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            if model_name in predictions:
                y_pred = predictions[model_name]
                
                # Create time index for plotting
                time_index = range(len(y_test))
                
                axes[i].plot(time_index, y_test, label='Actual', linewidth=2, color=self.colors[0])
                axes[i].plot(time_index, y_pred, label='Predicted', linewidth=2, color=self.colors[1])
                axes[i].set_title(f'{model_name} - Predictions vs Actual')
                axes[i].set_ylabel('Price ($)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.xlabel('Time Steps')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_performance(self, model_scores: dict, save_path: str = None) -> None:
        """
        Plot model performance comparison.
        
        Args:
            model_scores (dict): Dictionary of model evaluation scores
            save_path (str): Path to save the plot
        """
        models = list(model_scores.keys())
        metrics = ['r2', 'rmse', 'mae', 'mape']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [model_scores[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 15, save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance (pd.DataFrame): Feature importance data
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
        """
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors[0])
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}', va='center', ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot correlation matrix of features.
        
        Args:
            data (pd.DataFrame): Data with features
            save_path (str): Path to save the plot
        """
        # Select numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, data: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot distribution of daily returns.
        
        Args:
            data (pd.DataFrame): Stock data with daily returns
            save_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(data['Daily_Return'].dropna(), bins=50, alpha=0.7, color=self.colors[0])
        ax1.set_title('Distribution of Daily Returns')
        ax1.set_xlabel('Daily Return')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(data['Daily_Return'].dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Daily Returns')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns distribution plot saved to {save_path}")
        
        plt.show()

def main():
    """
    Example usage of the StockVisualizer.
    """
    from data_collector import StockDataCollector
    from data_preprocessor import StockDataPreprocessor
    
    # Collect and preprocess data
    collector = StockDataCollector("AAPL")
    data = collector.get_stock_data(period="1y", interval="1d")
    
    preprocessor = StockDataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data)
    
    # Create visualizations
    visualizer = StockVisualizer()
    
    # Plot price data
    visualizer.plot_price_data(data, "AAPL")
    
    # Plot technical indicators
    visualizer.plot_technical_indicators(processed_data['original_data'], "AAPL")
    
    # Plot correlation matrix
    visualizer.plot_correlation_matrix(processed_data['original_data'])
    
    # Plot returns distribution
    visualizer.plot_returns_distribution(processed_data['original_data'])

if __name__ == "__main__":
    main()
