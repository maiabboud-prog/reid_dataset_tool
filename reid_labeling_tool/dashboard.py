"""
Dashboard module - adds 3-step workflow without modifying core labeling tool
"""
import json
from pathlib import Path
from flask import render_template, request, jsonify, redirect
import threading
import time
import sys
import gc

# Import config to get CLUSTER_PATH
from config import CLUSTER_PATH, BASE_PATH

# Import extraction and clustering functions
from extraction import extract_features, get_image_paths, is_valid_image, extraction_state
from clustering import load_features_and_paths, cluster_faiss_kmeans, cluster_agglomerative, save_clusters

DASHBOARD_CONFIG = Path(__file__).parent / "dashboard_config.json"

# Clustering state
clustering_state = {
    "running": False, 
    "progress": 0, 
    "message": "", 
    "clusters": 0, 
    "error": None,
    "start_time": None,
    "elapsed": 0
}

def load_config():
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

def save_config(config):
    with open(DASHBOARD_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)

def register_dashboard_routes(app):
    """Register dashboard routes without affecting existing routes."""
    
    @app.route('/dashboard')
    def dashboard():
        config = load_config()
        return render_template('dashboard.html', config=config)
    
    @app.route('/dashboard/extract')
    def dashboard_extract():
        config = load_config()
        return render_template('extract.html', config=config)
    
    @app.route('/dashboard/cluster')
    def dashboard_cluster():
        config = load_config()
        return render_template('cluster.html', config=config)
    
    @app.route('/dashboard/label')
    def dashboard_label():
        """Redirect to labeling page with the clustered dataset"""
        config = load_config()
        dataset_path = config['steps']['labeling']['dataset_path']
        
        if not dataset_path or dataset_path == "":
            dataset_path = str(CLUSTER_PATH)
        
        # Also ensure features are loaded
        from features import load_features
        load_features()
        
        config['steps']['labeling']['status'] = 'in_progress'
        save_config(config)
        
        return redirect('/')
    
    @app.route('/api/dashboard/extract/start', methods=['POST'])
    def api_dashboard_extract_start():
        global extraction_state
        
        if extraction_state['running']:
            return jsonify({'success': False, 'error': 'Extraction already running'}), 400
        
        data = request.json
        model_path = data.get('model_path', '')
        dataset_path = data.get('dataset_path', '')
        output_dir = data.get('output_dir', '')
        batch_size = int(data.get('batch_size', 32))
        min_size = int(data.get('min_size', 20))
        
        if not model_path or not dataset_path or not output_dir:
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        # Make paths absolute if relative
        model_path = str(Path(model_path))
        dataset_path = str(Path(dataset_path))
        output_dir = str(Path(output_dir))
        
        config = load_config()
        config['steps']['extraction'].update({
            'model_path': model_path,
            'dataset_path': dataset_path,
            'output_dir': output_dir,
            'status': 'in_progress'
        })
        save_config(config)
        
        extraction_state.update({
            'running': True,
            'progress': 0,
            'current': 0,
            'total': 0,
            'valid': 0,
            'skipped': 0,
            'error': None,
            'message': 'Starting extraction...'
        })
        
        def run_extraction():
            global extraction_state
            try:
                result = extract_features(model_path, dataset_path, output_dir, batch_size, min_size)
                
                if isinstance(result, dict):
                    if result.get('success'):
                        extraction_state['progress'] = 100
                        extraction_state['message'] = 'Complete!'
                        config = load_config()
                        config['steps']['extraction']['status'] = 'completed'
                        
                        # Store paths for clustering step
                        features_file = result.get('features_file')
                        paths_file = result.get('paths_file')
                        meta_file = result.get('meta_file')
                        
                        if features_file and paths_file:
                            # Make paths absolute
                            features_file = str(Path(features_file).absolute())
                            paths_file = str(Path(paths_file).absolute())
                            if meta_file:
                                meta_file = str(Path(meta_file).absolute())
                            
                            config['steps']['clustering']['features_file'] = features_file
                            config['steps']['clustering']['paths_file'] = paths_file
                            if meta_file:
                                config['steps']['clustering']['meta_file'] = meta_file
                            
                            # Also update labeling features
                            config['steps']['labeling']['features_dat'] = features_file
                            config['steps']['labeling']['features_json'] = paths_file
                            if meta_file:
                                config['steps']['labeling']['meta_file'] = meta_file
                        
                        save_config(config)
                    else:
                        extraction_state['error'] = result.get('error', 'Unknown error')
                        extraction_state['message'] = 'Failed'
                        config = load_config()
                        config['steps']['extraction']['status'] = 'failed'
                        save_config(config)
                else:
                    extraction_state['error'] = 'Invalid return type from extract_features'
                    extraction_state['message'] = 'Failed'
                    
            except Exception as e:
                extraction_state['error'] = str(e)
                extraction_state['message'] = 'Failed'
                config = load_config()
                config['steps']['extraction']['status'] = 'failed'
                save_config(config)
            finally:
                extraction_state['running'] = False
                extraction_state['message'] = 'Done'
        
        thread = threading.Thread(target=run_extraction)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/extract/status')
    def api_dashboard_extract_status():
        return jsonify({
            'running': extraction_state['running'],
            'percent': extraction_state['progress'],
            'processed': extraction_state['current'],
            'total': extraction_state['total'],
            'valid': extraction_state['valid'],
            'skipped': extraction_state['skipped'],
            'error': extraction_state['error'],
            'message': extraction_state['message']
        })
    
    @app.route('/api/dashboard/extract/cancel', methods=['POST'])
    def api_dashboard_extract_cancel():
        global extraction_state
        extraction_state['running'] = False
        config = load_config()
        config['steps']['extraction']['status'] = 'cancelled'
        save_config(config)
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/cluster/save_config', methods=['POST'])
    def api_dashboard_cluster_save_config():
        """Save clustering configuration without starting clustering"""
        data = request.json
        
        # Get all clustering parameters
        features_file = data.get('features_file', '')
        paths_file = data.get('paths_file', '')
        meta_file = data.get('meta_file', '')
        output_dir = data.get('output_dir', '')
        algorithm = data.get('algorithm', 'faiss')
        parameters = data.get('parameters', {})
        
        # Validate required fields
        if not features_file or not paths_file or not output_dir:
            return jsonify({
                'success': False, 
                'error': 'Missing required parameters: features_file, paths_file, and output_dir are required'
            }), 400
        
        # Convert to absolute paths
        features_file = str(Path(features_file).absolute())
        paths_file = str(Path(paths_file).absolute())
        if meta_file:
            meta_file = str(Path(meta_file).absolute())
        output_dir = str(Path(output_dir).absolute())
        
        # Load current config
        config = load_config()
        
        # Update clustering configuration
        config['steps']['clustering'].update({
            'features_file': features_file,
            'paths_file': paths_file,
            'meta_file': meta_file,
            'output_dir': output_dir,
            'method': algorithm,
            'parameters': parameters,
            'status': 'configured'  # Mark as configured but not started
        })
        
        # ===== FIX: Sync to Labeling Page =====
        # Update labeling features reference
        config['steps']['labeling']['features_dat'] = features_file
        config['steps']['labeling']['features_json'] = paths_file
        if meta_file:
            config['steps']['labeling']['meta_file'] = meta_file
        
        # CRITICAL: Sync the output directory to labeling dataset path
        config['steps']['labeling']['dataset_path'] = output_dir
        
        # Update labeling status to indicate data is ready
        # Check if output_dir has person_ folders
        output_path = Path(output_dir)
        if output_path.exists() and output_path.is_dir():
            person_folders = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('person_')]
            if person_folders:
                config['steps']['labeling']['status'] = 'ready'
            else:
                config['steps']['labeling']['status'] = 'configured'  # No clusters yet
        else:
            config['steps']['labeling']['status'] = 'configured'
        
        # Save config
        save_config(config)
        
        print(f"✅ Clustering configuration saved and synced to labeling:")
        print(f"   Features: {features_file}")
        print(f"   Paths: {paths_file}")
        print(f"   Output: {output_dir}")
        print(f"   Labeling dataset: {config['steps']['labeling']['dataset_path']}")
        print(f"   Labeling status: {config['steps']['labeling']['status']}")
        
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/cluster/start', methods=['POST'])
    def api_dashboard_cluster_start():
        global clustering_state
        
        if clustering_state['running']:
            return jsonify({'success': False, 'error': 'Clustering already running'}), 400
        
        data = request.json
        features_file = data.get('features_file', '')
        paths_file = data.get('paths_file', '')
        meta_file = data.get('meta_file', '')
        algorithm = data.get('algorithm', 'agg')
        parameters = data.get('parameters', {})
        output_dir = data.get('output_dir', '')
        
        if not features_file or not paths_file or not output_dir:
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        # Make paths absolute
        features_file = str(Path(features_file).absolute())
        paths_file = str(Path(paths_file).absolute())
        if meta_file:
            meta_file = str(Path(meta_file).absolute())
        output_dir = str(Path(output_dir).absolute())
        
        config = load_config()
        config['steps']['clustering'].update({
            'method': algorithm,
            'parameters': parameters,
            'output_dir': output_dir,
            'features_file': features_file,
            'paths_file': paths_file,
            'meta_file': meta_file,
            'status': 'in_progress'
        })
        save_config(config)
        
        clustering_state.update({
            'running': True,
            'progress': 0,
            'message': 'Loading features...',
            'clusters': 0,
            'error': None,
            'start_time': time.time(),
            'elapsed': 0
        })
        
        def run_clustering():
            global clustering_state
            try:
                clustering_state['message'] = 'Loading features...'
                clustering_state['progress'] = 10
                
                if meta_file and Path(meta_file).exists():
                    features, image_paths = load_features_and_paths(features_file, paths_file, meta_file)
                else:
                    features, image_paths = load_features_and_paths(features_file, paths_file)
                
                clustering_state['progress'] = 20
                clustering_state['message'] = f'Loaded {len(features)} features'
                
                clustering_state['message'] = 'Clustering...'
                clustering_state['progress'] = 30
                
                n = len(features)
                print(f"📊 Clustering {n} images with algorithm: {algorithm}")
                print(f"📂 Features from: {features_file}")
                print(f"📂 Output to: {output_dir}")
                
                if algorithm == 'faiss':
                    num_clusters = int(parameters.get('k', 250))
                    seed = int(parameters.get('seed', 42))
                    iterations = int(parameters.get('iterations', 100))
                    use_gpu = parameters.get('gpu', False)
                    
                    labels = cluster_faiss_kmeans(features, num_clusters, seed, iterations, use_gpu)
                    method_name = f"FAISS K-Means (k={num_clusters})"
                    
                elif algorithm == 'agg':
                    threshold = float(parameters.get('threshold', 0.6))
                    linkage = parameters.get('linkage', 'complete')
                    affinity = parameters.get('affinity', 'cosine')
                    
                    if n > 10000:
                        clustering_state['message'] = f'Clustering with sampling (5000 images)...'
                        print(f"⚠️ Large dataset ({n} images). Using sampling for Agglomerative...")
                        labels = cluster_agglomerative(features, threshold, linkage, affinity, max_samples=5000)
                    else:
                        labels = cluster_agglomerative(features, threshold, linkage, affinity)
                    method_name = f"Agglomerative (threshold={threshold})"
                    
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                clustering_state['progress'] = 60
                clustering_state['clusters'] = len(set(labels))
                clustering_state['message'] = f'Found {clustering_state["clusters"]} clusters'
                
                clustering_state['message'] = 'Saving clusters...'
                clustering_state['progress'] = 70
                
                output_path = save_clusters(image_paths, labels, output_dir, method_name)
                
                clustering_state['progress'] = 100
                clustering_state['message'] = 'Complete!'
                clustering_state['elapsed'] = time.time() - clustering_state['start_time']
                
                # Update config with cluster output path
                config = load_config()
                config['steps']['clustering']['status'] = 'completed'
                config['steps']['labeling']['dataset_path'] = output_path
                config['steps']['labeling']['status'] = 'ready'
                
                # Ensure features are linked to labeling step
                config['steps']['labeling']['features_dat'] = features_file
                config['steps']['labeling']['features_json'] = paths_file
                if meta_file:
                    config['steps']['labeling']['meta_file'] = meta_file
                
                save_config(config)
                
                # Reload features for labeling
                from features import load_features
                load_features()
                
                print(f"✅ Clustering completed in {clustering_state['elapsed']:.2f}s! Output: {output_path}")
                print(f"📂 Labeling dataset: {output_path}")
                print(f"📂 Features file: {features_file}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                clustering_state['error'] = str(e)
                clustering_state['message'] = f'Failed: {str(e)[:100]}'
                config = load_config()
                config['steps']['clustering']['status'] = 'failed'
                save_config(config)
            finally:
                clustering_state['running'] = False
                gc.collect()
        
        thread = threading.Thread(target=run_clustering)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/cluster/status')
    def api_dashboard_cluster_status():
        return jsonify({
            'running': clustering_state['running'],
            'percent': clustering_state['progress'],
            'message': clustering_state['message'],
            'clusters': clustering_state['clusters'],
            'error': clustering_state['error'],
            'elapsed': clustering_state.get('elapsed', 0)
        })
    
    @app.route('/api/dashboard/cluster/cancel', methods=['POST'])
    def api_dashboard_cluster_cancel():
        global clustering_state
        clustering_state['running'] = False
        config = load_config()
        config['steps']['clustering']['status'] = 'cancelled'
        save_config(config)
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/ensure_features_loaded', methods=['POST'])
    def api_dashboard_ensure_features_loaded():
        """Ensure features are loaded for the labeling page"""
        from features import load_features
        
        try:
            # Force reload features from current config
            load_features()
            return jsonify({'success': True, 'loaded': True})
        except Exception as e:
            print(f"Error loading features: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/dashboard/config/save', methods=['POST'])
    def api_dashboard_config_save():
        data = request.json
        step = data.get('step')
        step_data = data.get('data', {})
        config = load_config()
        config['steps'][step].update(step_data)
        if step != 'labeling':
            config['steps'][step]['status'] = 'not_started'
        save_config(config)
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/config/load')
    def api_dashboard_config_load():
        return jsonify(load_config())
    
    @app.route('/api/dashboard/set_labeling_path', methods=['POST'])
    def api_dashboard_set_labeling_path():
        data = request.json
        dataset_path = data.get('dataset_path', '')
        
        # Clean the path - convert backslashes to forward slashes
        dataset_path = dataset_path.replace('\\', '/')
        
        # Check if the path exists
        if dataset_path and not Path(dataset_path).exists():
            return jsonify({'success': False, 'error': f'Path does not exist: {dataset_path}'}), 400
        
        # Check if it has person_ folders
        if dataset_path:
            path_obj = Path(dataset_path)
            if path_obj.exists() and path_obj.is_dir():
                person_folders = [d.name for d in path_obj.iterdir() if d.is_dir() and d.name.startswith('person_')]
                if not person_folders:
                    return jsonify({'success': False, 'error': 'No person_ folders found. Please check the path.'}), 400
        
        config = load_config()
        config['steps']['labeling']['dataset_path'] = dataset_path
        config['steps']['labeling']['status'] = 'in_progress'
        save_config(config)
        
        print(f"📂 Labeling dataset updated to: {dataset_path}")
        
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/test_path', methods=['POST'])
    def api_dashboard_test_path():
        data = request.json
        path_str = data.get('path', '')
        
        # Normalize path
        path_str = path_str.replace('\\', '/')
        path_obj = Path(path_str)
        
        result = {
            'original': data.get('path', ''),
            'normalized': path_str,
            'exists': path_obj.exists(),
            'is_dir': path_obj.is_dir() if path_obj.exists() else False,
            'absolute': str(path_obj.absolute()),
            'parent': str(path_obj.parent)
        }
        
        if path_obj.exists() and path_obj.is_dir():
            person_folders = [d.name for d in path_obj.iterdir() if d.is_dir() and d.name.startswith('person_')]
            result['person_folders'] = person_folders[:10]
            result['person_folders_count'] = len(person_folders)
        
        return jsonify(result)
    
    @app.route('/api/dashboard/set_features', methods=['POST'])
    def api_dashboard_set_features():
        """Set the features files for labeling"""
        data = request.json
        features_dat = data.get('features_dat', '')
        features_json = data.get('features_json', '')
        meta_file = data.get('meta_file', '')
        
        if not features_dat or not features_json:
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        # Validate files exist
        if not Path(features_dat).exists():
            return jsonify({'success': False, 'error': f'Features DAT file not found: {features_dat}'}), 400
        if not Path(features_json).exists():
            return jsonify({'success': False, 'error': f'Features JSON file not found: {features_json}'}), 400
        
        # Update config with new paths
        from config import update_features_paths
        update_features_paths(features_dat, features_json, meta_file)
        
        # Reload features
        from features import load_features
        load_features()
        
        return jsonify({'success': True})
    
    @app.route('/api/dashboard/current_status')
    def api_dashboard_current_status():
        """Get current status of all steps"""
        config = load_config()
        return jsonify({
            'extraction': {
                'status': config['steps']['extraction']['status'],
                'output_dir': config['steps']['extraction'].get('output_dir', '')
            },
            'clustering': {
                'status': config['steps']['clustering']['status'],
                'output_dir': config['steps']['clustering'].get('output_dir', ''),
                'features_file': config['steps']['clustering'].get('features_file', '')
            },
            'labeling': {
                'status': config['steps']['labeling']['status'],
                'dataset_path': config['steps']['labeling'].get('dataset_path', ''),
                'features_dat': config['steps']['labeling'].get('features_dat', '')
            }
        })