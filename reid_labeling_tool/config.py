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
    
    # First priority: Try to get from clustering step (the features used for clustering)
    clustering_features = config.get('steps', {}).get('clustering', {}).get('features_file', '')
    clustering_paths = config.get('steps', {}).get('clustering', {}).get('paths_file', '')
    
    # Second priority: Try from extraction step
    extraction_output = config.get('steps', {}).get('extraction', {}).get('output_dir', '')
    
    if clustering_features and clustering_paths:
        features_dat = Path(clustering_features)
        features_json = Path(clustering_paths)
        if features_dat.exists() and features_json.exists():
            print(f"📂 Using features from clustering: {features_dat}")
            return features_dat, features_json
    
    # If clustering step has output_dir, look for features there
    if extraction_output:
        output_dir = Path(extraction_output)
        if not output_dir.is_absolute():
            output_dir = BASE_PATH / output_dir
        features_dat = output_dir / "features_dataset.dat"
        features_json = output_dir / "features_dataset_paths.json"
        if features_dat.exists() and features_json.exists():
            print(f"📂 Using features from extraction output: {features_dat}")
            return features_dat, features_json
    
    # Then try from labeling step
    labeling_features = config.get('steps', {}).get('labeling', {}).get('features_dat', '')
    labeling_paths = config.get('steps', {}).get('labeling', {}).get('features_json', '')
    
    if labeling_features and labeling_paths:
        features_dat = Path(labeling_features)
        features_json = Path(labeling_paths)
        if features_dat.exists() and features_json.exists():
            print(f"📂 Using features from labeling: {features_dat}")
            return features_dat, features_json
    
    # Return defaults
    print(f"📂 Using default features: {FEATURES_DAT}")
    return FEATURES_DAT, FEATURES_JSON

def update_features_paths(features_dat, features_json, meta_file=None):
    """Update feature paths in dashboard config."""
    config = load_dashboard_config()
    
    # Update in clustering step (where features were used)
    if 'clustering' not in config['steps']:
        config['steps']['clustering'] = {}
    config['steps']['clustering']['features_file'] = str(features_dat)
    config['steps']['clustering']['paths_file'] = str(features_json)
    if meta_file:
        config['steps']['clustering']['meta_file'] = str(meta_file)
    
    # Also update in labeling step for reference
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

def get_meta_file():
    """Get the meta file path from dashboard config."""
    config = load_dashboard_config()
    
    # Try from clustering first
    meta_file = config.get('steps', {}).get('clustering', {}).get('meta_file', '')
    if meta_file and Path(meta_file).exists():
        return Path(meta_file)
    
    # Try from extraction output
    extraction_output = config.get('steps', {}).get('extraction', {}).get('output_dir', '')
    if extraction_output:
        output_dir = Path(extraction_output)
        if not output_dir.is_absolute():
            output_dir = BASE_PATH / output_dir
        meta_file = output_dir / "features_dataset_meta.json"
        if meta_file.exists():
            return meta_file
    
    # Try from labeling
    meta_file = config.get('steps', {}).get('labeling', {}).get('meta_file', '')
    if meta_file and Path(meta_file).exists():
        return Path(meta_file)
    
    return None

def get_cluster_output_dir():
    """Get the cluster output directory from dashboard config."""
    config = load_dashboard_config()
    
    # Try from clustering step
    cluster_output = config.get('steps', {}).get('clustering', {}).get('output_dir', '')
    if cluster_output:
        path_obj = Path(cluster_output)
        if not path_obj.is_absolute():
            path_obj = BASE_PATH / path_obj
        if path_obj.exists():
            return path_obj
    
    # Try from labeling step
    labeling_path = config.get('steps', {}).get('labeling', {}).get('dataset_path', '')
    if labeling_path:
        path_obj = Path(labeling_path)
        if not path_obj.is_absolute():
            path_obj = BASE_PATH / path_obj
        if path_obj.exists():
            return path_obj
    
    return CLUSTER_PATH

# ===== COMPATIBILITY FUNCTIONS FOR routes.py and features.py =====

def load_state():
    """Load state from dashboard config (compatibility function)."""
    return load_dashboard_config()

def save_state(config):
    """Save state to dashboard config (compatibility function)."""
    save_dashboard_config(config)

def get_cluster_path():
    """Get the current cluster path (compatibility function)."""
    return get_cluster_output_dir()

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