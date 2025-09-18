# Stock Analytics Architecture

## Project Structure

```
stockaroo/
├── stockaroo/                 # Main package
│   ├── __init__.py           # Package initialization and exports
│   ├── cli.py               # Command line interface
│   ├── data/                # Data collection and preprocessing
│   │   ├── __init__.py
│   │   ├── collector.py     # Yahoo Finance data collection
│   │   └── preprocessor.py  # Data preprocessing and feature engineering
│   ├── models/              # Machine learning models
│   │   ├── __init__.py
│   │   └── predictor.py     # ML models and evaluation
│   ├── ui/                  # User interfaces
│   │   ├── __init__.py
│   │   └── streamlit_app.py # Web dashboard
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   └── visualizer.py    # Data visualization
│   └── config/              # Configuration management
│       ├── __init__.py
│       └── settings.py      # Application settings
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_data_collector.py
│   └── test_models.py
├── docs/                    # Documentation
│   └── ARCHITECTURE.md
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
├── README.md               # Project documentation
├── run_ui.py              # UI launcher (legacy)
├── simple_main.py         # Simple CLI script (legacy)
├── demo.py                # Demo script (legacy)
└── main.py                # Legacy main script
```

## Architecture Principles

### 1. Separation of Concerns
- **Data Layer**: Handles data collection and preprocessing
- **Model Layer**: Contains machine learning models and evaluation
- **UI Layer**: Provides user interfaces (CLI and Web)
- **Utils Layer**: Shared utilities and visualization
- **Config Layer**: Centralized configuration management

### 2. Modularity
Each module is self-contained with clear interfaces:
- `StockDataCollector`: Fetches data from Yahoo Finance
- `StockDataPreprocessor`: Creates features and prepares data
- `StockPredictor`: Trains and evaluates ML models
- `StockVisualizer`: Creates charts and visualizations

### 3. Configuration Management
Centralized configuration through `Settings` class:
- Model parameters
- Data collection settings
- UI preferences
- Environment variables

### 4. Error Handling
- Graceful error handling in all modules
- Logging for debugging and monitoring
- Validation of inputs and data

### 5. Testing
- Unit tests for each module
- Mock external dependencies
- Test coverage for critical paths

## Module Dependencies

```
config/
├── settings.py (no dependencies)

data/
├── collector.py (yfinance, pandas, numpy)
└── preprocessor.py (pandas, numpy, sklearn)

models/
└── predictor.py (sklearn, numpy, pandas)

utils/
└── visualizer.py (matplotlib, seaborn, plotly)

ui/
└── streamlit_app.py (streamlit, plotly, all other modules)

cli.py (all modules)
```

## Data Flow

1. **Data Collection**: `StockDataCollector` fetches raw stock data
2. **Preprocessing**: `StockDataPreprocessor` creates features and splits data
3. **Model Training**: `StockPredictor` trains ML models
4. **Evaluation**: Models are evaluated on test data
5. **Visualization**: Results are displayed via `StockVisualizer` or Streamlit UI

## Configuration

The application uses a hierarchical configuration system:

```python
from stockaroo.config import get_settings

settings = get_settings()
print(settings.data.default_symbol)  # "AAPL"
print(settings.model.available_models)  # ["linear_regression", ...]
```

## CLI Usage

```bash
# Analyze a stock
stockaroo analyze AAPL --period 2y --models linear_regression lasso

# Launch web UI
stockaroo ui --port 8501

# Show version
stockaroo version
```

## Web UI

The Streamlit web interface provides:
- Interactive parameter controls
- Real-time data visualization
- Model comparison
- Export capabilities

## Testing

Run tests with:
```bash
pytest tests/
```

## Installation

Development installation:
```bash
pip install -e .
```

Production installation:
```bash
pip install stockaroo
```
