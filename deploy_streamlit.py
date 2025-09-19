#!/usr/bin/env python3
"""
Deployment script for Streamlit Cloud
This script ensures the package is properly installed and imports work correctly.
"""

import sys
import os
import subprocess

def install_package():
    """Install the stockaroo package in development mode."""
    try:
        # Install the package in development mode
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ Package installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install package: {e}")
        return False

def test_imports():
    """Test that all imports work correctly."""
    try:
        from stockaroo.data.collector import StockDataCollector
        from stockaroo.models.predictor import StockPredictor
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main deployment function."""
    print("🚀 Starting Streamlit Cloud deployment setup...")
    
    # Install package
    if not install_package():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    print("✅ Deployment setup completed successfully!")
    print("📱 You can now run: streamlit run stockaroo/ui/streamlit_app.py")

if __name__ == "__main__":
    main()
