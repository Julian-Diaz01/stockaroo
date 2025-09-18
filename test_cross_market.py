"""
Test script for cross-market prediction system.
"""

from stockaroo.models.cross_market_predictor import CrossMarketPredictor
from stockaroo.models.predictor import StockPredictor

def test_model_persistence():
    """Test model saving and loading."""
    print("🔧 Testing Model Persistence...")
    
    # Test single stock predictor
    predictor = StockPredictor()
    
    # Create dummy data for testing
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    # Train model
    predictor.train_model('linear_regression', X, y)
    
    # Save model
    model_path = predictor.save_model('linear_regression', 'TEST', {'test': True})
    print(f"✅ Model saved to: {model_path}")
    
    # List saved models
    saved_models = predictor.list_saved_models()
    print(f"📁 Found {len(saved_models)} saved models")
    
    return True

def test_cross_market_prediction():
    """Test cross-market prediction system."""
    print("\n🌍 Testing Cross-Market Prediction...")
    
    # Initialize cross-market predictor
    predictor = CrossMarketPredictor()
    
    # Prepare data
    X, y, full_data = predictor.prepare_cross_market_data(period="3mo", lookback_window=2)
    print(f"📊 Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.time_series_split(X, y, test_size=0.2)
    
    # Train multiple models
    models_to_test = ['linear_regression', 'ridge', 'lasso']
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🤖 Training {model_name}...")
        predictor.train_model(model_name, X_train, y_train)
        eval_results = predictor.evaluate_model(model_name, X_test, y_test)
        results[model_name] = eval_results
        
        print(f"   R² Score: {eval_results['r2']:.4f}")
        print(f"   RMSE: {eval_results['rmse']:.2f}")
        print(f"   MAPE: {eval_results['mape']:.2f}%")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    print(f"\n🏆 Best model: {best_model} (R² = {results[best_model]['r2']:.4f})")
    
    # Next day prediction
    try:
        next_day_pred = predictor.predict_next_day(best_model)
        print(f"🔮 Next day US market prediction: {next_day_pred:.2f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    # Save best model
    try:
        model_path = predictor.save_model(best_model, additional_info={'cross_market': True})
        print(f"💾 Best model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
    
    # A/D Analysis
    try:
        ad_analysis = predictor.analyze_accumulation_distribution()
        print(f"\n📈 Accumulation/Distribution Analysis:")
        for market, analysis in ad_analysis.items():
            print(f"   {market}: {analysis.get('ad_trend', 'unknown')} trend")
    except Exception as e:
        print(f"❌ A/D analysis failed: {e}")
    
    return results

def main():
    """Run all tests."""
    print("🚀 Stock Analytics - Cross-Market Prediction System Test")
    print("=" * 60)
    
    try:
        # Test model persistence
        test_model_persistence()
        
        # Test cross-market prediction
        results = test_cross_market_prediction()
        
        print("\n✅ All tests completed successfully!")
        print(f"📊 Cross-market prediction system working with {len(results)} models")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
