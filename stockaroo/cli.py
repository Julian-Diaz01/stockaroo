"""
Command Line Interface for Stock Analytics.
"""

import argparse
import sys
import logging
from typing import Optional

from .data.collector import StockDataCollector
from .models.predictor import StockPredictor
from .config import get_settings


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_stock(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    models: Optional[list] = None,
    output_file: Optional[str] = None
) -> None:
    """Run stock analysis from command line."""
    
    if models is None:
        models = ["linear_regression", "lasso"]
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting analysis for {symbol}")
    
    try:
        # Collect data
        collector = StockDataCollector(symbol)
        data = collector.get_stock_data(period=period, interval=interval)
        
        # Prepare data using improved method
        predictor = StockPredictor()
        X, y = predictor.prepare_time_series_data(data, lookback_window=10, prediction_horizon=1)
        X_train, X_test, y_train, y_test = predictor.time_series_split(X, y, test_size=0.2)
        
        # Train and evaluate models
        results = {}
        for model_name in models:
            try:
                predictor.train_model(model_name, X_train, y_train)
                eval_results = predictor.evaluate_model(model_name, X_test, y_test)
                results[model_name] = eval_results
                
                logger.info(f"{model_name}: R²={eval_results['r2']:.4f}, RMSE={eval_results['rmse']:.2f}")
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
        
        # Save results if output file specified
        if output_file:
            import pandas as pd
            results_df = pd.DataFrame([
                {
                    'Model': model,
                    'R2_Score': results[model]['r2'],
                    'RMSE': results[model]['rmse'],
                    'MAE': results[model]['mae'],
                    'MAPE': results[model]['mape']
                }
                for model in results.keys()
            ])
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        # Find best model
        if results:
            best_model = max(results.keys(), key=lambda x: results[x]['r2'])
            logger.info(f"Best model: {best_model} (R²={results[best_model]['r2']:.4f})")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Analytics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stockaroo analyze AAPL
  stockaroo analyze MSFT --period 2y --models linear_regression lasso
  stockaroo analyze GOOGL --output results.csv
  stockaroo ui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a stock')
    analyze_parser.add_argument('symbol', help='Stock symbol (e.g., AAPL)')
    analyze_parser.add_argument('--period', default='1y', 
                              choices=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                              help='Data period (default: 1y)')
    analyze_parser.add_argument('--interval', default='1d',
                              choices=['1d', '1wk', '1mo'],
                              help='Data interval (default: 1d)')
    analyze_parser.add_argument('--models', nargs='+',
                              choices=['linear_regression', 'ridge', 'lasso', 'random_forest'],
                              default=['linear_regression', 'lasso'],
                              help='Models to use (default: linear_regression lasso)')
    analyze_parser.add_argument('--output', '-o', help='Output CSV file')
    analyze_parser.add_argument('--log-level', default='INFO',
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              help='Logging level (default: INFO)')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch web interface')
    ui_parser.add_argument('--port', type=int, default=8501, help='Port number (default: 8501)')
    ui_parser.add_argument('--advanced', action='store_true', help='Launch advanced UI with cross-market analysis')
    ui_parser.add_argument('--host', default='localhost', help='Host (default: localhost)')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        setup_logging(args.log_level)
        analyze_stock(
            symbol=args.symbol,
            period=args.period,
            interval=args.interval,
            models=args.models,
            output_file=args.output
        )
    
    elif args.command == 'ui':
        try:
            import subprocess
            import sys
            
            # Choose UI app based on advanced flag
            app_file = "stockaroo/ui/advanced_streamlit_app.py" if args.advanced else "stockaroo/ui/streamlit_app.py"
            
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                app_file,
                "--server.port", str(args.port),
                "--server.address", args.host
            ])
        except ImportError:
            print("Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)
    
    elif args.command == 'version':
        from . import __version__
        print(f"Stock Analytics v{__version__}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
