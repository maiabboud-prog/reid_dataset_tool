
"""
This script contains common utility functions for loading data and saving clusters.
"""
import json
from pathlib import Path
from shutil import copy2
from collections import defaultdict
import numpy as np

def load_features_and_paths(features_file, paths_file, meta_file):
    """
    Loads features, image paths, and metadata from files.
    """
    print(f"📂 Loading metadata from '{meta_file}'...")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    
    num_images = meta["num_images"]
    feature_dim = meta["feature_dim"]
    
    print(f"📊 Total images: {num_images}")
    print(f"📐 Feature dimension: {feature_dim}")
    
    print(f"📂 Loading features from '{features_file}'...")
    features = np.memmap(features_file, dtype=np.float32, mode="r", shape=(num_images, feature_dim))
    
    # Copy to RAM for processing (FAISS and other libraries might need contiguous memory)
    print("📥 Copying features to RAM...")
    features = np.array(features)
    
    print(f"📂 Loading paths from '{paths_file}'...")
    with open(paths_file, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
        
    if len(image_paths) != num_images:
        raise ValueError(f"Mismatch: {len(image_paths)} paths vs {num_images} features")
        
    print(f"✅ Loaded {len(image_paths)} images with {feature_dim}-dim features")
    
    return features, image_paths

def save_clusters(image_paths, labels, output_dir, method_name="Clustering"):
    """
    Save images organized by cluster folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:  # Skip noise points
            continue
        clusters[label].append((idx, image_paths[idx]))
    
    print(f"Saving {len(clusters)} clusters...")
    
    for label, items in sorted(clusters.items()):
        cluster_folder = output_dir / f"person_{label:04d}"
        cluster_folder.mkdir(exist_ok=True)
        
        for idx, path in items:
            path = Path(path)
            filename = path.name
            dest = cluster_folder / filename
            
            if dest.exists():
                dest = cluster_folder / f"{path.stem}_{idx}{path.suffix}"
            
            copy2(path, dest)
        
        print(f"  Person {label:04d}: {len(items)} images")
        
    # Save statistics
    stats_file = output_dir / "cluster_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"{method_name} Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Number of clusters found: {len(clusters)}\n")
        
        num_noise = sum(1 for l in labels if l == -1)
        if num_noise > 0:
            f.write(f"Number of noise points: {num_noise}\n")
        
        f.write("\n")
        
        for label, items in sorted(clusters.items()):
            f.write(f"Person {label:04d}: {len(items)} images\n")
            
    print(f"💾 Clusters saved to '{output_dir}/'")

