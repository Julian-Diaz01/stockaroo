"""
Stock Analytics Package

A comprehensive Python package for stock analysis and prediction using machine learning.
"""

__version__ = "1.0.0"
__author__ = "Stock Analytics Team"
__email__ = "contact@stockaroo.com"

# Package imports
from .data.collector import StockDataCollector
from .models.predictor import StockPredictor
from .utils.visualizer import StockVisualizer

__all__ = [
    "StockDataCollector",
    "StockPredictor",
    "StockVisualizer"
]
