"""
New main entry point using the restructured package.
"""

import sys
import logging
from stockaroo import StockDataCollector, StockDataPreprocessor, ImprovedStockPredictor
from stockaroo.config import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function using the new package structure."""
    logger.info("Starting Stock Analytics Project (New Architecture)")
    logger.info("=" * 60)
    
    # Get settings
    settings = get_settings()
    
    # Configuration
    SYMBOL = settings.data.default_symbol
    PERIOD = settings.data.default_period
    INTERVAL = settings.data.default_interval
    MODELS = settings.model.available_models[:3]  # Use first 3 models
    
    logger.info(f"Using settings: {SYMBOL}, {PERIOD}, {INTERVAL}")
    logger.info(f"Models: {MODELS}")
    
    try:
        # Step 1: Data Collection
        logger.info("Step 1: Collecting stock data...")
        collector = StockDataCollector(SYMBOL)
        raw_data = collector.get_stock_data(period=PERIOD, interval=INTERVAL)
        
        # Get company information
        company_info = collector.get_company_info()
        logger.info(f"Company: {company_info.get('name', 'N/A')}")
        logger.info(f"Sector: {company_info.get('sector', 'N/A')}")
        logger.info(f"Data collected: {len(raw_data)} records")
        logger.info(f"Date range: {raw_data.index.min().date()} to {raw_data.index.max().date()}")
        
        # Step 2: Data Preprocessing
        logger.info("\nStep 2: Preprocessing data...")
        preprocessor = StockDataPreprocessor()
        processed_data = preprocessor.preprocess_pipeline(
            raw_data, 
            target_col='Close', 
            horizon=1,
            test_size=settings.data.default_test_size
        )
        
        logger.info(f"Features created: {len(processed_data['feature_names'])}")
        logger.info(f"Training samples: {processed_data['X_train'].shape[0]}")
        logger.info(f"Test samples: {processed_data['X_test'].shape[0]}")
        
        # Step 3: Model Training and Evaluation
        logger.info("\nStep 3: Training and evaluating models...")
        predictor = ImprovedStockPredictor()
        
        # Train and evaluate selected models
        eval_results = {}
        for model_name in MODELS:
            try:
                predictor.train_model(model_name, processed_data['X_train'], processed_data['y_train'])
                eval_results[model_name] = predictor.evaluate_model(
                    model_name, processed_data['X_test'], processed_data['y_test']
                )
                logger.info(f"{model_name}: R²={eval_results[model_name]['r2']:.4f}")
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
        
        # Get best model
        if eval_results:
            best_model = max(eval_results.keys(), key=lambda x: eval_results[x]['r2'])
            logger.info(f"\nBest model: {best_model}")
            logger.info(f"R² Score: {eval_results[best_model]['r2']:.4f}")
            logger.info(f"RMSE: ${eval_results[best_model]['rmse']:.2f}")
            logger.info(f"MAPE: {eval_results[best_model]['mape']:.2f}%")
        
        logger.info("\n✅ Analysis completed successfully!")
        logger.info("🚀 Use 'stockaroo ui' to launch the web interface")
        logger.info("📊 Use 'stockaroo analyze SYMBOL' for CLI analysis")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
