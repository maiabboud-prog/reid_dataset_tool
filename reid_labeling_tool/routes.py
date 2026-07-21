"""
Flask routes for the ReID Labeling Tool
"""
import os
import json
import shutil
import io
from datetime import datetime
from flask import request, jsonify, send_from_directory, send_file
import numpy as np

from config import get_cluster_path, load_state, save_state, get_current_cluster_path
from features import get_feature_lookup

def get_cluster_images(cluster_dir):
    """Helper to get sorted image names from a directory."""
    if not cluster_dir.is_dir():
        return []
    return sorted([f.name for f in cluster_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

def register_routes(app):
    """Register all routes with the Flask app."""
    
    @app.route('/')
    def index():
        from flask import render_template
        # Get the current cluster path
        cluster_path = get_current_cluster_path()
        return render_template('index.html', cluster_path=str(cluster_path))
    
    @app.route('/images/<cluster>/<filename>')
    def serve_image(cluster, filename):
        cluster_path = get_current_cluster_path()
        return send_from_directory(str(cluster_path / cluster), filename)
    
    @app.route('/preview/<cluster>')
    def preview_image(cluster):
        """Return first image from cluster for preview"""
        from config import get_current_cluster_path
        cluster_path = get_current_cluster_path()
        cluster_dir = cluster_path / cluster
        images = get_cluster_images(cluster_dir)
        if images:
            try:
                return send_from_directory(str(cluster_dir), images[0])
            except Exception as e:
                print(f"Error serving preview: {e}")
                return '', 404
        return '', 404
    
    @app.route('/api/clusters')
    def get_clusters():
        cluster_path = get_current_cluster_path()
        clusters = {}
        if not cluster_path.exists():
            return jsonify(clusters)
        for d in cluster_path.iterdir():
            if d.is_dir() and d.name.startswith('person_'):
                count = len(get_cluster_images(d))
                clusters[d.name] = count
        return jsonify(clusters)
    
    @app.route('/api/cluster/<name>')
    def get_cluster(name):
        cluster_path = get_current_cluster_path()
        images = get_cluster_images(cluster_path / name)
        return jsonify({'images': images})
    
    @app.route('/api/new_cluster', methods=['POST'])
    def new_cluster():
        cluster_path = get_current_cluster_path()
        data = request.json
        cluster_id = data.get('cluster_id')
        if not cluster_id:
            return jsonify({'success': False, 'error': 'Cluster ID is required'}), 400
        
        target_dir = cluster_path / cluster_id
        if target_dir.exists():
            return jsonify({'success': False, 'error': 'Cluster already exists'}), 400
            
        target_dir.mkdir(exist_ok=True)
        return jsonify({'success': True})
    
    @app.route('/api/move', methods=['POST'])
    def move():
        cluster_path = get_current_cluster_path()
        data = request.json
        src, target, filename = data.get('source_cluster'), data.get('target_cluster'), data.get('filename')
        
        if not all([src, target, filename]):
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        src_path = cluster_path / src / filename
        target_dir = cluster_path / target
        target_path = target_dir / filename
        
        if not src_path.exists():
            return jsonify({'success': False, 'error': f'Source file not found: {src_path}'}), 404
        
        target_dir.mkdir(exist_ok=True)
        if target_path.exists():
            base, ext = os.path.splitext(filename)
            target_path = target_dir / f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
        
        shutil.move(str(src_path), str(target_path))
        
        src_dir = cluster_path / src
        if src != target and not any(src_dir.iterdir()):
            try:
                src_dir.rmdir()
            except OSError as e:
                print(f"Error removing directory {src_dir}: {e}")

        return jsonify({'success': True, 'target': target})
    
    @app.route('/api/merge', methods=['POST'])
    def merge():
        cluster_path = get_current_cluster_path()
        data = request.json
        clusters, target = data.get('clusters'), data.get('target')
        
        if not all([clusters, target]) or len(clusters) < 1:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        target_dir = cluster_path / target
        target_dir.mkdir(exist_ok=True)
        
        for src_name in set(clusters):
            if src_name == target:
                continue
            src_dir = cluster_path / src_name
            if src_dir.exists():
                for img in src_dir.iterdir():
                    if img.is_file():
                        dst = target_dir / img.name
                        if dst.exists():
                            base, ext = os.path.splitext(img.name)
                            dst = target_dir / f"{base}_m_{datetime.now().strftime('%H%M%S')}{ext}"
                        shutil.move(str(img), str(dst))
                if not any(src_dir.iterdir()):
                    try:
                        src_dir.rmdir()
                    except OSError as e:
                        print(f"Error removing directory {src_dir}: {e}")

        return jsonify({'success': True})
    
    @app.route('/api/split', methods=['POST'])
    def split():
        cluster_path = get_current_cluster_path()
        data = request.json
        src, target, images = data.get('source_cluster'), data.get('target_cluster'), data.get('images')

        if not all([src, target, images]):
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        target_dir = cluster_path / target
        target_dir.mkdir(exist_ok=True)
        
        for img_name in images:
            src_file = cluster_path / src / img_name
            if src_file.exists():
                dst = target_dir / img_name
                if dst.exists():
                    base, ext = os.path.splitext(img_name)
                    dst = target_dir / f"{base}_s_{datetime.now().strftime('%H%M%S')}{ext}"
                shutil.move(str(src_file), str(dst))

        src_dir = cluster_path / src
        if not any(src_dir.iterdir()):
            try:
                src_dir.rmdir()
            except OSError as e:
                print(f"Error removing directory {src_dir}: {e}")

        return jsonify({'success': True})
    
    @app.route('/api/export')
    def export():
        cluster_path = get_current_cluster_path()
        labels = {}
        if not cluster_path.exists():
            return jsonify(labels)
            
        for d in cluster_path.iterdir():
            if d.is_dir() and d.name.startswith('person_'):
                for f in d.iterdir():
                    if f.is_file():
                        labels[f.name] = d.name
        
        output = io.BytesIO(json.dumps(labels, indent=2).encode('utf-8'))
        output.seek(0)
        return send_file(output, mimetype='application/json', 
                        as_attachment=True, download_name=f'labels_{datetime.now().strftime("%Y%m%d")}.json')
    
    @app.route('/api/find_similar', methods=['POST'])
    def find_similar():
        """Find the top 5 most similar clusters based on selected images."""
        feature_lookup = get_feature_lookup()
        
        if not feature_lookup:
            return jsonify({'success': False, 'error': 'Feature lookup is not available.'}), 500

        data = request.json
        source_cluster = data.get('cluster')
        image_names = data.get('images')

        if not all([source_cluster, image_names]):
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        print(f"\n🔍 Finding similar images for {len(image_names)} selection in '{source_cluster}'...")

        query_vectors = []
        for img_name in image_names:
            # Try exact match first
            if img_name in feature_lookup:
                query_vectors.append(feature_lookup[img_name])
                print(f"   ✅ Found exact match for: {img_name}")
            else:
                # If no exact match, try adding a .jpg extension (common case)
                img_name_jpg = f"{img_name}.jpg"
                if img_name_jpg in feature_lookup:
                    query_vectors.append(feature_lookup[img_name_jpg])
                    print(f"   ✅ Found match by adding .jpg: {img_name_jpg}")
                else:
                    # Fallback: check if any key starts with the image name
                    base_name = img_name.split('.')[0]
                    found = False
                    for key in feature_lookup.keys():
                        if key.startswith(base_name):
                            query_vectors.append(feature_lookup[key])
                            print(f"   ✅ Found partial match: {key} for {img_name}")
                            found = True
                            break
                    if not found:
                        print(f"   ❌ No feature vector found for: {img_name}")
        
        if not query_vectors:
            return jsonify({'success': False, 'error': 'None of the selected images have feature vectors.'}), 404

        query_feature = np.mean(query_vectors, axis=0)
        query_feature /= np.linalg.norm(query_feature)

        # Get current cluster path
        cluster_path = get_current_cluster_path()
        
        similarities = []
        for cluster_dir in cluster_path.iterdir():
            if not cluster_dir.is_dir() or not cluster_dir.name.startswith('person_'):
                continue
            
            if cluster_dir.name == source_cluster:
                continue

            cluster_vectors = []
            image_files = get_cluster_images(cluster_dir)
            if not image_files:
                continue

            for img_name in image_files:
                if img_name in feature_lookup:
                    cluster_vectors.append(feature_lookup[img_name])

            if not cluster_vectors:
                continue

            # Calculate similarity
            scores = np.dot(cluster_vectors, query_feature)
            max_sim = np.max(scores)
            best_match_idx = np.argmax(scores)
            best_match_image = image_files[best_match_idx]

            if max_sim > -1.0:
                similarities.append({
                    "cluster": cluster_dir.name,
                    "score": float(max_sim),
                    "count": len(image_files),
                    "best_match": best_match_image
                })

        # Sort by score (descending) and take top 5
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the array directly (matching the minimal code)
        return jsonify(similarities[:5])


    @app.route('/api/move_multiple', methods=['POST'])
    def move_multiple():
        cluster_path = get_current_cluster_path()
        data = request.json
        src, target, filenames = data.get('source_cluster'), data.get('target_cluster'), data.get('filenames')
        
        if not all([src, target, filenames]) or not isinstance(filenames, list):
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        target_dir = cluster_path / target
        target_dir.mkdir(exist_ok=True)
        moved_count = 0
        
        for filename in filenames:
            src_path = cluster_path / src / filename
            if src_path.exists():
                target_path = target_dir / filename
                if target_path.exists():
                    base, ext = os.path.splitext(filename)
                    target_path = target_dir / f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
                shutil.move(str(src_path), str(target_path))
                moved_count += 1

        src_dir = cluster_path / src
        if src != target and not any(src_dir.iterdir()):
            try:
                src_dir.rmdir()
            except OSError as e:
                print(f"Error removing directory {src_dir}: {e}")

        return jsonify({'success': True, 'moved': moved_count, 'target': target})