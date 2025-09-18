"""
Tests for machine learning models.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from stockaroo.models import StockPredictor


class TestStockPredictor:
    """Test cases for StockPredictor."""
    
    def test_init(self):
        """Test predictor initialization."""
        predictor = StockPredictor()
        assert 'linear_regression' in predictor.models
        assert 'ridge' in predictor.models
        assert 'lasso' in predictor.models
        assert 'random_forest' in predictor.models
        assert len(predictor.trained_models) == 0
        assert len(predictor.model_scores) == 0
    
    def test_train_model(self):
        """Test model training."""
        predictor = StockPredictor()
        
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        
        result = predictor.train_model('linear_regression', X_train, y_train)
        
        assert 'linear_regression' in predictor.trained_models
        assert result['model_name'] == 'linear_regression'
        assert 'train_score' in result
        assert result['train_score'] >= 0
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        predictor = StockPredictor()
        
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(20, 10)
        y_test = np.random.rand(20)
        
        # Train first
        predictor.train_model('linear_regression', X_train, y_train)
        
        # Then evaluate
        result = predictor.evaluate_model('linear_regression', X_test, y_test)
        
        assert 'linear_regression' in predictor.model_scores
        assert 'r2' in result
        assert 'rmse' in result
        assert 'mae' in result
        assert 'mape' in result
        assert 'predictions' in result
    
    def test_evaluate_model_not_trained(self):
        """Test evaluation of untrained model."""
        predictor = StockPredictor()
        
        X_test = np.random.rand(20, 10)
        y_test = np.random.rand(20)
        
        with pytest.raises(ValueError, match="not trained yet"):
            predictor.evaluate_model('linear_regression', X_test, y_test)
    
    def test_get_best_model(self):
        """Test getting best model."""
        predictor = StockPredictor()
        
        # Mock some scores
        predictor.model_scores = {
            'model1': {'r2': 0.8},
            'model2': {'r2': 0.9},
            'model3': {'r2': 0.7}
        }
        
        best_model = predictor.get_best_model()
        assert best_model == 'model2'
    
    def test_get_best_model_no_scores(self):
        """Test getting best model with no scores."""
        predictor = StockPredictor()
        
        with pytest.raises(ValueError, match="No models have been evaluated"):
            predictor.get_best_model()
