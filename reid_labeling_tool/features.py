"""
Feature loading and management
"""
import numpy as np
import json
import os
import traceback
from pathlib import Path
from config import get_features_paths, update_features_paths

feature_lookup = {}

def load_features():
    """Load and normalize features from current paths."""
    global feature_lookup
    
    # Get current feature paths from config
    features_dat, features_json = get_features_paths()
    
    print("\n--- Loading Feature Vectors ---")
    print(f"Attempting to load features from:")
    print(f"  - DAT File: {features_dat}")
    print(f"  - JSON Path File: {features_json}")
    
    if not features_dat.exists():
        print(f"❌ Error: DAT file not found at the specified path.")
        feature_lookup = {}
        return
    if not features_json.exists():
        print(f"❌ Error: JSON file not found at the specified path.")
        feature_lookup = {}
        return

    try:
        # Load paths first to know the expected number of features
        with open(features_json, 'r') as f:
            paths = json.load(f)
        num_paths = len(paths)
        print(f"Found {num_paths} image paths in JSON file.")

        # Load raw features
        features_raw = np.fromfile(str(features_dat), dtype=np.float32)
        print(f"Raw features loaded from .dat file, total elements: {features_raw.size}")
        
        # Determine feature dimension and reshape
        if num_paths > 0 and features_raw.size % num_paths == 0:
            feature_dim = features_raw.size // num_paths
            features = features_raw.reshape(num_paths, feature_dim)
            print(f"Reshaped features to: {features.shape} (num_images, feature_dim)")
        else:
            error_msg = (
                f"Cannot reshape features. The number of elements in the DAT file ({features_raw.size}) "
                f"is not divisible by the number of paths in the JSON file ({num_paths})."
            )
            print(f"❌ {error_msg}")
            # Try to find a common dimension if reshaping fails
            for dim in [2048, 1024, 512]:
                if features_raw.size % dim == 0:
                    num_images = features_raw.size // dim
                    if num_images == num_paths:
                         features = features_raw.reshape(num_images, dim)
                         print(f"Fallback reshape successful with dim={dim}")
                         break
            else:
                raise ValueError(error_msg)

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9  # Avoid division by zero
        features_normalized = features / norms

        # Create the lookup dictionary
        temp_lookup = {}
        for i, path in enumerate(paths):
            filename = os.path.basename(path)
            temp_lookup[filename] = features_normalized[i]
        
        feature_lookup = temp_lookup
        print(f"✅ Successfully loaded and normalized {len(feature_lookup)} feature vectors.")
        if feature_lookup:
            print(f"   Sample filenames: {list(feature_lookup.keys())[:5]}")
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to decode JSON from {features_json}. It might be corrupted. Details: {e}")
        feature_lookup = {}
    except ValueError as e:
        print(f"❌ Error during feature reshaping: {e}")
        feature_lookup = {}
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading features: {e}")
        feature_lookup = {}
        traceback.print_exc()

def get_feature_lookup():
    """Return the feature lookup dictionary."""
    return feature_lookup