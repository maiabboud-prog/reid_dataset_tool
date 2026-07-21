
"""
Script 1: Extract OSNet features and save them incrementally.

Memory-efficient version:
- Does not keep all features in RAM
- Uses numpy.memmap
- Saves image paths separately
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(
    0,
    'C:\\Users\\Manar\\Documents\\REID\\Cluster_id'
)
sys.path.insert(
    0,
    'C:\\Users\\Manar\\Documents\\REID\\Cluster_id\\features'
)

from reid_labeling_tool.extractor_base_cpu import FeatureExtraction


# ========== CONFIG ==========
DATASET_FOLDER = "2026041316"
INPUT_DIR = f"clustered_persons/dataset/{DATASET_FOLDER}"

FEATURES_FILE = f"features_dataset_hour_{DATASET_FOLDER}.dat"
PATHS_FILE = f"features_dataset_hour_paths_{DATASET_FOLDER}.json"
META_FILE = f"features_dataset_hour_meta_{DATASET_FOLDER}.json"

QUALITY_MIN_SIZE = 20

# Number of images processed at once
BATCH_SIZE = 32

# =============================


def get_image_paths(input_dir):
    """Get all image paths."""

    image_extensions = {
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.tiff'
    }

    image_paths = []

    for path in Path(input_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_paths.append(path)

    return image_paths


def is_valid_image(img_path):
    """Check whether image passes the quality filter."""

    img = cv2.imread(str(img_path))

    if img is None:
        return False

    h, w = img.shape[:2]

    if h < QUALITY_MIN_SIZE or w < QUALITY_MIN_SIZE:
        return False

    return True


def extract_features_and_save(input_dir):
    """Extract OSNet features incrementally."""

    print("Loading OSNet extractor...")
    extractor = FeatureExtraction()

    print("Searching for images...")
    image_paths = get_image_paths(input_dir)

    print(f"📁 Found {len(image_paths)} images")

    # --------------------------------------------------
    # First pass: count valid images
    # --------------------------------------------------

    print("Checking image quality...")

    valid_paths = []
    skipped = 0

    for i, img_path in enumerate(image_paths):

        if is_valid_image(img_path):
            valid_paths.append(str(img_path))
        else:
            skipped += 1

        if (i + 1) % 1000 == 0:
            print(
                f"Checked {i + 1}/{len(image_paths)} images"
            )

    num_images = len(valid_paths)

    print(f"✅ Valid images: {num_images}")
    print(f"⚠️ Skipped images: {skipped}")

    if num_images == 0:
        print("❌ No valid images found")
        return

    # --------------------------------------------------
    # Detect feature dimension using one image
    # --------------------------------------------------

    print("Detecting feature dimension...")

    sample_img = cv2.imread(valid_paths[0])

    sample_feat = extractor.predict_img(sample_img)[0]
    sample_feat = np.asarray(sample_feat).flatten()

    feature_dim = sample_feat.shape[0]

    print(f"Feature dimension: {feature_dim}")

    # --------------------------------------------------
    # Create disk-backed memory map
    # --------------------------------------------------

    print("Creating memory-mapped feature file...")

    features = np.memmap(
        FEATURES_FILE,
        dtype=np.float32,
        mode='w+',
        shape=(num_images, feature_dim)
    )

    # --------------------------------------------------
    # Extract and save incrementally
    # --------------------------------------------------

    print("Extracting features...")

    batch_images = []
    batch_indices = []

    for i, img_path in enumerate(valid_paths):

        img = cv2.imread(img_path)

        batch_images.append(img)
        batch_indices.append(i)

        # Process batch
        if (
            len(batch_images) == BATCH_SIZE
            or i == num_images - 1
        ):

            for image, index in zip(
                batch_images,
                batch_indices
            ):

                feat = extractor.predict_img(image)[0]

                feat = np.asarray(feat).flatten()

                # L2 normalization
                norm = np.linalg.norm(feat)

                if norm > 0:
                    feat = feat / norm

                # Write directly to disk
                features[index] = feat

            # Flush changes to disk
            features.flush()

            print(
                f"Processed {i + 1}/{num_images} images"
            )

            # Clear batch
            batch_images.clear()
            batch_indices.clear()

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    meta = {
        "num_images": num_images,
        "feature_dim": feature_dim,
        "dtype": "float32",
        "features_file": FEATURES_FILE
    }

    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            valid_paths,
            f,
            indent=2
        )

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            indent=2
        )

    # Close memory map
    del features

    print("\n🎉 Feature extraction completed!")
    print(f"💾 Features saved to: {FEATURES_FILE}")
    print(f"💾 Paths saved to: {PATHS_FILE}")
    print(f"💾 Metadata saved to: {META_FILE}")
    print(f"📐 Feature dimension: {feature_dim}")
    print(f"🖼️ Number of images: {num_images}")


if __name__ == "__main__":
    extract_features_and_save(INPUT_DIR)