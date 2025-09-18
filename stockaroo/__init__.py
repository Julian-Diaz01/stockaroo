"""
Stock Analytics Package

A comprehensive Python package for stock analysis and prediction using machine learning.
"""

__version__ = "1.0.0"
__author__ = "Stock Analytics Team"
__email__ = "contact@stockaroo.com"

# Package imports
from .data.collector import StockDataCollector
from .data.multi_market_collector import MultiMarketCollector
from .models.predictor import StockPredictor
from .models.cross_market_predictor import CrossMarketPredictor
from .utils.visualizer import StockVisualizer

__all__ = [
    "StockDataCollector",
    "MultiMarketCollector",
    "StockPredictor",
    "CrossMarketPredictor",
    "StockVisualizer"
]
