"""
Clustering functionality for the ReID Labeling Tool
"""
import numpy as np
import faiss
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path
import shutil
import os
import json
from datetime import datetime
import sys
import gc

def load_features_and_paths(features_file, paths_file, meta_file=None):
    """Load features and paths from files"""
    # Convert to Path objects
    features_file = Path(features_file)
    paths_file = Path(paths_file)
    
    # Check if files exist
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file.absolute()}")
    if not paths_file.exists():
        raise FileNotFoundError(f"Paths file not found: {paths_file.absolute()}")
    
    print(f"📂 Loading features from: {features_file}")
    print(f"📂 Loading paths from: {paths_file}")
    
    # Load features
    features = np.fromfile(str(features_file), dtype=np.float32)
    print(f"Raw features size: {len(features)}")
    
    # Try to load from meta file first
    feature_dim = None
    num_images = None
    
    if meta_file and Path(meta_file).exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            feature_dim = meta.get('feature_dim', 512)
            num_images = meta.get('num_images', 0)
            print(f"📋 Meta: dimension={feature_dim}, images={num_images}")
    
    # Reshape features
    if feature_dim and num_images:
        try:
            features = features.reshape(num_images, feature_dim)
            print(f"✅ Reshaped using meta: {features.shape}")
        except ValueError as e:
            print(f"⚠️ Could not reshape using meta: {e}")
            # Try auto-detect
            for dim in [512, 1024, 2048, 4096]:
                if len(features) % dim == 0:
                    features = features.reshape(-1, dim)
                    print(f"✅ Auto-reshaped to dimension: {dim}")
                    break
            else:
                raise ValueError(f"Cannot reshape features of size {len(features)}")
    else:
        # Try to auto-detect
        for dim in [512, 1024, 2048, 4096]:
            if len(features) % dim == 0:
                features = features.reshape(-1, dim)
                print(f"✅ Auto-reshaped to dimension: {dim}")
                break
        else:
            raise ValueError(f"Cannot reshape features of size {len(features)}")
    
    # Load paths - handle both list and dict formats
    try:
        with open(paths_file, 'r') as f:
            image_paths_data = json.load(f)
        
        # Convert to list if it's a dict
        if isinstance(image_paths_data, dict):
            # If it's a dict with keys like "0", "1", etc.
            if all(str(i) in image_paths_data for i in range(len(image_paths_data))):
                image_paths = [image_paths_data[str(i)] for i in range(len(image_paths_data))]
            else:
                # If it's a dict with other keys, just take values
                image_paths = list(image_paths_data.values())
            print(f"✅ Converted dict to list: {len(image_paths)} paths")
        else:
            image_paths = image_paths_data
            print(f"✅ Loaded list: {len(image_paths)} paths")
            
    except Exception as e:
        raise ValueError(f"Error loading paths file: {e}")
    
    print(f"✅ Loaded {len(image_paths)} image paths")
    
    # Verify matching counts
    if len(features) != len(image_paths):
        print(f"⚠️ Warning: Features ({len(features)}) and paths ({len(image_paths)}) count mismatch")
        # Truncate to the smaller size
        min_len = min(len(features), len(image_paths))
        features = features[:min_len]
        image_paths = image_paths[:min_len]
        print(f"Truncated to {min_len} samples")
    
    return features, image_paths

def cluster_faiss_kmeans(features, num_clusters, seed=42, iterations=100, use_gpu=False):
    """Cluster using FAISS k-means"""
    features = features.astype(np.float32)
    n, d = features.shape
    
    print(f"\n🔮 Clustering {n} images into {num_clusters} clusters using FAISS K-Means...")
    print(f"Feature dimension: {d}")
    
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0):
        print("Normalizing features...")
        features = features / norms
    
    # Cluster all points
    kmeans = faiss.Kmeans(
        d=d,
        k=num_clusters,
        niter=iterations,
        verbose=True,
        seed=seed,
        gpu=use_gpu,
    )
    kmeans.train(features)
    _, labels = kmeans.index.search(features, 1)
    labels = labels.flatten()
    
    num_clusters_found = len(set(labels))
    print(f"✅ Found {num_clusters_found} clusters.")
    
    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}")
    
    return labels
def cluster_agglomerative(features, threshold=0.6, linkage='complete', affinity='cosine', max_samples=5000):
    """Cluster using Agglomerative Clustering with sampling for large datasets"""
    n, d = features.shape
    print(f"\n🔮 Clustering {n} images with Agglomerative (threshold: {threshold})...")
    print(f"Feature dimension: {d}")
    
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0):
        print("Normalizing features...")
        features = features / norms
    
    # For large datasets, use sampling
    if n > max_samples:
        print(f"⚠️ Large dataset ({n} images). Using random sample of {max_samples}...")
        np.random.seed(42)
        indices = np.random.choice(n, max_samples, replace=False)
        sample_features = features[indices]
        
        print("Computing distance matrix for sample...")
        if affinity == 'cosine':
            dist_matrix = cosine_distances(sample_features)
        else:
            from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
            if affinity == 'euclidean':
                dist_matrix = euclidean_distances(sample_features)
            elif affinity == 'manhattan':
                dist_matrix = manhattan_distances(sample_features)
            else:
                dist_matrix = cosine_distances(sample_features)
        
        dist_matrix = np.clip(dist_matrix, 0, 2)
        
        print("Fitting Agglomerative Clustering on sample...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=linkage,
            distance_threshold=threshold
        )
        
        sample_labels = clustering.fit_predict(dist_matrix)
        
        # Assign all points to nearest cluster (using centroids)
        print("Assigning all points to nearest cluster...")
        # Compute centroids of sample clusters
        centroids = []
        for label in set(sample_labels):
            cluster_features = sample_features[sample_labels == label]
            if len(cluster_features) > 0:
                centroid = np.mean(cluster_features, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                centroids.append(centroid)
        centroids = np.array(centroids)
        
        # Assign all points to nearest centroid
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(features, centroids)
        labels = np.argmax(similarities, axis=1)
        
    else:
        # For smaller datasets, compute full distance matrix
        print("Computing distance matrix...")
        if affinity == 'cosine':
            dist_matrix = cosine_distances(features)
        else:
            from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
            if affinity == 'euclidean':
                dist_matrix = euclidean_distances(features)
            elif affinity == 'manhattan':
                dist_matrix = manhattan_distances(features)
            else:
                dist_matrix = cosine_distances(features)
        
        dist_matrix = np.clip(dist_matrix, 0, 2)
        
        print("Fitting Agglomerative Clustering...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=linkage,
            distance_threshold=threshold
        )
        
        labels = clustering.fit_predict(dist_matrix)
    
    num_clusters = len(set(labels))
    print(f"✅ Found {num_clusters} clusters.")
    
    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}")
    
    return labels

def save_clusters(image_paths, labels, output_dir, method_name):
    """Save clustered images to output directory"""
    output_dir = Path(output_dir)
    output_dir = output_dir.resolve()  # Convert to absolute path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving clusters to: {output_dir.absolute()}")
    
    # Create cluster directories
    cluster_dirs = {}
    saved_count = 0
    failed_count = 0
    
    for idx, label in enumerate(labels):
        cluster_id = f"person_{label:04d}"
        if cluster_id not in cluster_dirs:
            cluster_dir = output_dir / cluster_id
            cluster_dir.mkdir(exist_ok=True)
            cluster_dirs[cluster_id] = cluster_dir
        
        # Get the source path - handle different formats
        try:
            src_path_str = image_paths[idx]
            # If it's a list or tuple, get the first element
            if isinstance(src_path_str, (list, tuple)):
                src_path_str = src_path_str[0]
            src_path = Path(src_path_str)
            
            # If the path is relative, try to resolve it relative to the current directory
            if not src_path.is_absolute():
                # Try to find the file by checking common locations
                possible_paths = [
                    src_path,  # Relative to current directory
                    Path("C:/Users/Manar/Documents/REID/Cluster_id") / src_path,  # Absolute with base
                    Path("C:/Users/Manar/Documents/REID/Cluster_id/dataset/2026041316") / src_path.name,  # In dataset folder
                ]
                
                src_path = None
                for p in possible_paths:
                    if p.exists():
                        src_path = p
                        break
                
                if src_path is None:
                    print(f"⚠️ Warning: Cannot find image: {src_path_str}")
                    failed_count += 1
                    continue
                    
        except (IndexError, KeyError, TypeError) as e:
            print(f"⚠️ Error accessing path at index {idx}: {e}")
            failed_count += 1
            continue
            
        if not src_path.exists():
            print(f"⚠️ Warning: Source image not found: {src_path}")
            failed_count += 1
            continue
            
        dst_path = cluster_dirs[cluster_id] / src_path.name
        
        # Handle duplicate filenames
        if dst_path.exists():
            base, ext = os.path.splitext(src_path.name)
            dst_path = cluster_dirs[cluster_id] / f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
        
        try:
            shutil.copy2(str(src_path), str(dst_path))
            saved_count += 1
        except Exception as e:
            print(f"⚠️ Error copying {src_path}: {e}")
            failed_count += 1
        
        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"Saved {idx + 1}/{len(labels)} images...")
    
    print(f"✅ Saved {saved_count} images, {failed_count} failed")
    
    # Save cluster summary
    summary_file = output_dir / "cluster_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Clustering Method: {method_name}\n")
        f.write(f"Total Images: {len(image_paths)}\n")
        f.write(f"Number of Clusters: {len(cluster_dirs)}\n\n")
        f.write("Cluster Distribution:\n")
        f.write("-" * 50 + "\n")
        
        for cluster_id, cluster_dir in sorted(cluster_dirs.items()):
            num_files = len(list(cluster_dir.glob("*")))
            f.write(f"{cluster_id}: {num_files} images\n")
    
    print(f"✅ Saved cluster summary to: {summary_file}")
    
    # Save cluster mapping as JSON
    mapping_file = output_dir / "cluster_mapping.json"
    mapping = {}
    for idx, label in enumerate(labels):
        cluster_id = f"person_{label:04d}"
        if cluster_id not in mapping:
            mapping[cluster_id] = []
        try:
            src_path_str = image_paths[idx]
            if isinstance(src_path_str, (list, tuple)):
                src_path_str = src_path_str[0]
            mapping[cluster_id].append(Path(src_path_str).name)
        except (IndexError, KeyError, TypeError):
            continue
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Saved cluster mapping to: {mapping_file}")
    
    return str(output_dir.absolute())