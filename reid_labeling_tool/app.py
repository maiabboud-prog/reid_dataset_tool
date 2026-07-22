"""
ReID Labeling Tool - Main Application Entry Point
Simple, fast, and clean
"""
from flask import Flask
import signal
import sys
from pathlib import Path

app = Flask(__name__)

# Increase timeout for long-running operations
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Import config FIRST to ensure paths are defined
from config import CLUSTER_PATH, get_current_cluster_path

# Import and register routes
from routes import register_routes
register_routes(app)

# Import and register dashboard routes
from dashboard import register_dashboard_routes
register_dashboard_routes(app)

# Import and load features
from features import load_features

if __name__ == '__main__':
    # Load features - this will now use dynamic paths from config
    load_features()
    
    # Get the current cluster path from config
    current_path = get_current_cluster_path()
    
    print(f"\n🚀 ReID Labeling Tool")
    if not current_path.exists():
        print(f"⚠️  Warning: Cluster path not found. Creating directory...")
        current_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Serving from: {current_path}")
    print(f"🌐 Dashboard: http://localhost:5000/dashboard")
    print(f"🌐 Labeling: http://localhost:5000/")
    print(f"🌐 http://localhost:5000\n")
    
    # Run with increased timeout
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)