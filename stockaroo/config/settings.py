"""
Configuration settings for the Stock Analytics application.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    
    # Available models
    available_models: List[str] = field(default_factory=lambda: [
        "linear_regression", "ridge", "lasso", "random_forest"
    ])
    
    # Model parameters
    random_forest_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": None,
        "min_samples_split": 2
    })
    
    ridge_params: Dict = field(default_factory=lambda: {
        "alpha": 1.0,
        "random_state": 42
    })
    
    lasso_params: Dict = field(default_factory=lambda: {
        "alpha": 0.1,
        "random_state": 42
    })


@dataclass
class DataConfig:
    """Configuration for data collection and preprocessing."""
    
    # Default data parameters
    default_symbol: str = "AAPL"
    default_period: str = "2y"
    default_interval: str = "1d"
    default_test_size: float = 0.2
    
    # Available periods
    available_periods: Dict[str, str] = field(default_factory=lambda: {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    })
    
    # Available intervals
    available_intervals: List[str] = field(default_factory=lambda: [
        "1d", "1wk", "1mo"
    ])
    
    # Technical indicators
    moving_averages: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])


@dataclass
class UIConfig:
    """Configuration for the user interface."""
    
    # Streamlit settings
    page_title: str = "Stock Analytics Dashboard"
    page_icon: str = "📈"
    layout: str = "wide"
    
    # Chart settings
    default_figsize: tuple = (12, 8)
    chart_height: int = 500
    
    # Colors
    colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'
    ])


@dataclass
class Settings:
    """Main application settings."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Application settings
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # API settings
    yahoo_finance_timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.debug:
            self.log_level = "DEBUG"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs) -> None:
    """Update global settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    
    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)


def reset_settings() -> None:
    """Reset global settings to defaults."""
    global _settings
    _settings = Settings()
