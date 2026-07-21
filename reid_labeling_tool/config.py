"""
Configuration settings for the ReID Labeling Tool
"""
from pathlib import Path
import json
import os

# ===== CONFIG =====
BASE_PATH = Path(r"C:\Users\Manar\Documents\REID\Cluster_id")
CLUSTER_PATH = BASE_PATH / "clustered_dataset" / "2026041316"

# Default feature paths (will be overridden by dashboard)
FEATURES_PATH = BASE_PATH / "features"
FEATURES_DAT = FEATURES_PATH / "features_dataset_hour.dat"
FEATURES_JSON = FEATURES_PATH / "features_dataset_hour_paths.json"

# Allowed image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Dashboard configuration file
DASHBOARD_CONFIG = Path(__file__).parent / "dashboard_config.json"

# State file for features (compatibility)
STATE_FILE = Path(__file__).parent / "state.json"

def load_dashboard_config():
    """Load dashboard configuration from file."""
    if DASHBOARD_CONFIG.exists():
        with open(DASHBOARD_CONFIG, 'r') as f:
            return json.load(f)
    return {
        "steps": {
            "extraction": {"status": "not_started", "model_path": "", "dataset_path": "", "output_dir": ""},
            "clustering": {"status": "not_started", "method": "agg", "parameters": {}, "output_dir": ""},
            "labeling": {"status": "not_started", "dataset_path": str(CLUSTER_PATH)}
        }
    }

def save_dashboard_config(config):
    """Save dashboard configuration to file."""
    with open(DASHBOARD_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)

def get_current_cluster_path():
    """Get the current cluster path from dashboard config."""
    config = load_dashboard_config()
    dataset_path = config.get('steps', {}).get('labeling', {}).get('dataset_path', '')
    if dataset_path:
        path_obj = Path(dataset_path)
        if not path_obj.is_absolute():
            path_obj = BASE_PATH / path_obj  # Make absolute
        if path_obj.exists():
            return path_obj
    return CLUSTER_PATH

def get_features_paths():
    """Get feature file paths from dashboard config or use defaults."""
    config = load_dashboard_config()
    
    # Try to get from labeling config
    features_dat = config.get('steps', {}).get('labeling', {}).get('features_dat', '')
    features_json = config.get('steps', {}).get('labeling', {}).get('features_json', '')
    
    if features_dat and features_json:
        return Path(features_dat), Path(features_json)
    
    # Try to get from clustering config (if features were extracted)
    features_dat = config.get('steps', {}).get('clustering', {}).get('features_file', '')
    features_json = config.get('steps', {}).get('clustering', {}).get('paths_file', '')
    
    if features_dat and features_json:
        return Path(features_dat), Path(features_json)
    
    # Return defaults
    return FEATURES_DAT, FEATURES_JSON

def update_features_paths(features_dat, features_json, meta_file=None):
    """Update feature paths in dashboard config."""
    config = load_dashboard_config()
    if 'labeling' not in config['steps']:
        config['steps']['labeling'] = {}
    config['steps']['labeling']['features_dat'] = str(features_dat)
    config['steps']['labeling']['features_json'] = str(features_json)
    if meta_file:
        config['steps']['labeling']['meta_file'] = str(meta_file)
    save_dashboard_config(config)
    
    # Update global variables
    global FEATURES_DAT, FEATURES_JSON
    FEATURES_DAT = Path(features_dat)
    FEATURES_JSON = Path(features_json)

# ===== COMPATIBILITY FUNCTIONS FOR routes.py and features.py =====

def load_state():
    """Load state from dashboard config (compatibility function)."""
    return load_dashboard_config()

def save_state(config):
    """Save state to dashboard config (compatibility function)."""
    save_dashboard_config(config)

def get_cluster_path():
    """Get the current cluster path (compatibility function)."""
    return get_current_cluster_path()

def load_features_state():
    """Load features state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_features_state(state):
    """Save features state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)