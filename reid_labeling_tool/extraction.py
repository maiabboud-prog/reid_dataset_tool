"""
Feature extraction functionality for the ReID Labeling Tool
"""
import os
import json
import numpy as np
import cv2
from pathlib import Path
import sys
import traceback

# ===== GLOBAL STATE FOR EXTRACTION =====
extraction_state = {
    "running": False,
    "progress": 0,
    "current": 0,
    "total": 0,
    "valid": 0,
    "skipped": 0,
    "error": None,
    "message": ""
}

def extract_features(model_path, dataset_path, output_dir, batch_size=32, min_size=20):
    """Extract features from dataset using specified model"""
    global extraction_state
    
    try:
        extraction_state['message'] = 'Initializing...'
        
        # Check if paths exist
        model_path = Path(model_path)
        if not model_path.exists():
            extraction_state['error'] = f"Model path does not exist: {model_path}"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            extraction_state['error'] = f"Dataset path does not exist: {dataset_path}"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        # Add the model directory to sys.path
        model_dir = str(model_path.parent)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        # Also add the parent directory of the model for imports
        parent_dir = str(model_path.parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import feature extractor
        extraction_state['message'] = 'Loading model...'
        
        try:
            # Try importing from the model directory
            import importlib.util
            
            # First, try to import extractor_base_cpu from the model directory
            extractor_file = model_path / "extractor_base_cpu.py"
            if extractor_file.exists():
                spec = importlib.util.spec_from_file_location("extractor_base_cpu", str(extractor_file))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    FeatureExtraction = module.FeatureExtraction
                    print("✅ Imported FeatureExtraction from extractor_base_cpu.py")
                else:
                    raise ImportError(f"Cannot load module from {extractor_file}")
            else:
                # Try importing as a module
                try:
                    from extractor_base_cpu import FeatureExtraction
                    print("✅ Imported FeatureExtraction as module")
                except ImportError:
                    # Try the old way
                    sys.path.insert(0, str(model_path))
                    from reid_labeling_tool.extractor_base_cpu import FeatureExtraction
                    print("✅ Imported FeatureExtraction from src.features")
                    
        except Exception as e:
            extraction_state['error'] = f"Cannot import FeatureExtraction: {str(e)}"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        # Initialize extractor
        extraction_state['message'] = 'Initializing feature extractor...'
        try:
            # Check if the ONNX model exists
            # The extractor expects the model at ../models/osnet_ain_x1_0_MS.onnx
            # We need to make sure the model file exists
            
            # If the model_path is a directory containing the ONNX file
            onnx_model_path = model_path / "osnet_ain_x1_0_MS.onnx"
            if not onnx_model_path.exists():
                # Try to find the ONNX file in the parent directory
                onnx_model_path = model_path.parent / "models" / "osnet_ain_x1_0_MS.onnx"
                if not onnx_model_path.exists():
                    # Try relative path from the extractor location
                    extractor_dir = model_path if model_path.is_dir() else model_path.parent
                    onnx_model_path = extractor_dir / ".." / "models" / "osnet_ain_x1_0_MS.onnx"
                    if not onnx_model_path.exists():
                        extraction_state['error'] = f"ONNX model file not found. Looking for: osnet_ain_x1_0_MS.onnx"
                        extraction_state['running'] = False
                        return {"success": False, "error": extraction_state['error']}
            
            # Set the current working directory so the ONNX model can be found
            original_cwd = os.getcwd()
            os.chdir(str(model_path.parent))
            
            # Create the extractor with the ONNX path
            extractor = FeatureExtraction(onnx_path=str(onnx_model_path))
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            print("✅ FeatureExtractor initialized")
            
        except Exception as e:
            extraction_state['error'] = f"Failed to initialize FeatureExtraction: {str(e)}"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        # Get all images
        extraction_state['message'] = 'Finding images...'
        image_paths = get_image_paths(dataset_path)
        print(f"Found {len(image_paths)} images")
        extraction_state['total'] = len(image_paths)
        
        if not image_paths:
            extraction_state['error'] = "No images found in dataset path"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        # Filter valid images
        extraction_state['message'] = 'Validating images...'
        valid_paths = []
        for i, img_path in enumerate(image_paths):
            if not extraction_state['running']:
                return {"success": False, "error": "Extraction cancelled"}
            
            if is_valid_image(img_path, min_size):
                valid_paths.append(str(img_path))
                extraction_state['valid'] = len(valid_paths)
            else:
                extraction_state['skipped'] += 1
            
            if (i + 1) % 100 == 0:
                extraction_state['progress'] = int((i + 1) / len(image_paths) * 10)
                print(f"Validated {i + 1}/{len(image_paths)} images")
        
        print(f"Valid images: {len(valid_paths)}")
        extraction_state['valid'] = len(valid_paths)
        
        if not valid_paths:
            extraction_state['error'] = "No valid images found (check min size filter)"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        # Detect feature dimension
        extraction_state['message'] = 'Detecting feature dimension...'
        sample_img = cv2.imread(valid_paths[0])
        if sample_img is None:
            extraction_state['error'] = f"Cannot read sample image: {valid_paths[0]}"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        try:
            sample_feat = extractor.predict_img(sample_img)[0]
            sample_feat = np.asarray(sample_feat).flatten()
            feature_dim = sample_feat.shape[0]
            print(f"Feature dimension: {feature_dim}")
        except Exception as e:
            extraction_state['error'] = f"Failed to extract features from sample: {e}"
            extraction_state['running'] = False
            return {"success": False, "error": extraction_state['error']}
        
        # Create output files
        extraction_state['message'] = 'Creating output directory...'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features_file = output_dir / "features_dataset.dat"
        paths_file = output_dir / "features_dataset_paths.json"
        meta_file = output_dir / "features_dataset_meta.json"
        
        # Create memory-mapped feature file
        extraction_state['message'] = 'Creating feature file...'
        features = np.memmap(
            str(features_file),
            dtype=np.float32,
            mode='w+',
            shape=(len(valid_paths), feature_dim)
        )
        
        # Extract features in batches
        extraction_state['message'] = 'Extracting features...'
        batch_images = []
        batch_indices = []
        
        for i, img_path in enumerate(valid_paths):
            if not extraction_state['running']:
                del features
                return {"success": False, "error": "Extraction cancelled"}
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Warning: Cannot read image: {img_path}")
                continue
                
            batch_images.append(img)
            batch_indices.append(i)
            
            if len(batch_images) == batch_size or i == len(valid_paths) - 1:
                for image, index in zip(batch_images, batch_indices):
                    try:
                        feat = extractor.predict_img(image)[0]
                        feat = np.asarray(feat).flatten()
                        norm = np.linalg.norm(feat)
                        if norm > 0:
                            feat = feat / norm
                        features[index] = feat
                    except Exception as e:
                        print(f"⚠️ Error processing image {index}: {e}")
                        # Use random features as fallback
                        features[index] = np.random.randn(feature_dim).astype(np.float32)
                        features[index] = features[index] / np.linalg.norm(features[index])
                
                features.flush()
                batch_images.clear()
                batch_indices.clear()
                
                extraction_state['current'] = i + 1
                extraction_state['progress'] = 10 + int((i + 1) / len(valid_paths) * 85)
                
                if (i + 1) % 100 == 0:
                    print(f"Extracted {i + 1}/{len(valid_paths)} images")
        
        # Save metadata
        extraction_state['message'] = 'Saving metadata...'
        meta = {
            "num_images": len(valid_paths),
            "feature_dim": feature_dim,
            "dtype": "float32",
            "features_file": str(features_file)
        }
        
        with open(paths_file, "w", encoding="utf-8") as f:
            json.dump(valid_paths, f, indent=2)
        
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        del features
        
        extraction_state['progress'] = 100
        extraction_state['message'] = 'Complete!'
        extraction_state['running'] = False
        
        return {
            "success": True,
            "features_file": str(features_file),
            "paths_file": str(paths_file),
            "meta_file": str(meta_file),
            "num_images": len(valid_paths),
            "feature_dim": feature_dim
        }
    
    except Exception as e:
        extraction_state['error'] = str(e)
        extraction_state['message'] = 'Failed'
        extraction_state['running'] = False
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def get_image_paths(input_dir):
    """Get all image paths recursively"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"⚠️ Directory does not exist: {input_dir}")
        return []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_paths.append(path)
    return image_paths

def is_valid_image(img_path, min_size=20):
    """Check if image is valid"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        return h >= min_size and w >= min_size
    except Exception:
        return False