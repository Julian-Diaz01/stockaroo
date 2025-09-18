"""
Launcher for the Advanced Streamlit UI with Cross-Market Analysis.
"""

import subprocess
import sys
import os

def main():
    """Launch the advanced Streamlit app."""
    print("🌍 Starting Advanced Stock Analytics Dashboard...")
    print("📊 Features:")
    print("   • Single Stock Analysis")
    print("   • Cross-Market Analysis (HK, EU, US)")
    print("   • Model Management")
    print("   • Accumulation/Distribution Analysis")
    print("🔗 The web interface will open in your browser.")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Run streamlit with the advanced app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "stockaroo/ui/advanced_streamlit_app.py",
            "--server.port", "8502"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Make sure you're in the project directory and have installed all dependencies.")

if __name__ == "__main__":
    main()
