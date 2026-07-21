"""
This script provides different clustering methods for person re-identification features.
"""
import numpy as np
import faiss
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from utils import load_features_and_paths, save_clusters
import argparse

# ========== CONFIG ==========
DATASET_FOLDER = "2026041316"
FEATURES_FILE = f"features/features_dataset_hour_{DATASET_FOLDER}.dat"
PATHS_FILE = f"features/features_dataset_hour_paths_{DATASET_FOLDER}.json"
META_FILE = f"features/features_dataset_hour_meta_{DATASET_FOLDER}.json"

# --- FAISS K-Means Config ---
FAISS_K = 250
FAISS_SEED = 42

# --- Agglomerative Config ---
AGG_DISTANCE_THRESHOLD = 0.6 # greater threshold means fewer clusters (more merging)

# =============================

def cluster_faiss_kmeans(features, num_clusters, seed=42):
    """
    Cluster features using FAISS k-means.
    """
    features = features.astype(np.float32)
    n, d = features.shape
    
    print(f"Clustering {n} images into {num_clusters} clusters using FAISS K-Means...")
    
    kmeans = faiss.Kmeans(
        d=d,
        k=num_clusters,
        niter=100,
        verbose=True,
        seed=seed,
        gpu=False,
    )
    
    kmeans.train(features)
    _, labels = kmeans.index.search(features, 1)
    labels = labels.flatten()
    
    print(f"✅ FAISS K-Means clustering complete! Found {len(set(labels))} clusters.")
    return labels

def cluster_agglomerative(features, threshold):
    """
    Cluster features using Agglomerative Clustering with cosine distance.
    """
    print(f"Clustering with Agglomerative Clustering (threshold: {threshold})...")
    print("Computing cosine distance matrix...")
    dist_matrix = cosine_distances(features)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='complete',
        distance_threshold=threshold
    )
    
    labels = clustering.fit_predict(dist_matrix)
    
    print(f"✅ Agglomerative clustering complete! Found {len(set(labels))} clusters.")
    return labels

def main():
    parser = argparse.ArgumentParser(description="Cluster person re-identification features.")
    parser.add_argument(
        '--method',
        type=str,
        default='agg',
        choices=['faiss', 'agg'],
        help="Clustering method to use: 'faiss' for K-Means or 'agg' for Agglomerative."
    )
    args = parser.parse_args()

    # Load data
    features, image_paths = load_features_and_paths(FEATURES_FILE, PATHS_FILE, META_FILE)

    if args.method == 'faiss':
        # FAISS K-Means Clustering
        labels = cluster_faiss_kmeans(features, num_clusters=FAISS_K, seed=FAISS_SEED)
        output_dir = f"processed_dataset/{DATASET_FOLDER}/faiss_{FAISS_K}"
        method_name = f"FAISS K-Means (k={FAISS_K})"

    elif args.method == 'agg':
        # Agglomerative Clustering
        labels = cluster_agglomerative(features, threshold=AGG_DISTANCE_THRESHOLD)
        output_dir = f"processed_dataset/{DATASET_FOLDER}/agg_complete_{AGG_DISTANCE_THRESHOLD}"
        method_name = f"Agglomerative (threshold={AGG_DISTANCE_THRESHOLD})"
    
    else:
        raise ValueError("Invalid clustering method specified.")

    # Save the clustered images
    save_clusters(image_paths, labels, output_dir, method_name)
    
    print("✅ All done!")

if __name__ == "__main__":
    main()
