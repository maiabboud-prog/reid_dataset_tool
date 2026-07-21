"""
ReID Labeling Tool - Optimized Version
Simple, fast, and clean
"""

import os, json, shutil, io
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory, send_file
import numpy as np

app = Flask(__name__)

# ===== CONFIG =====
BASE_PATH = Path(r"C:\Users\Manar\Documents\REID\Cluster_id")
CLUSTER_PATH = BASE_PATH / "processed_dataset" / "labeled_dataset" / "agg_complete_0.5_modified"
FEATURES_PATH = BASE_PATH / "features"
FEATURES_DAT = FEATURES_PATH / "features_dataset_hour.dat"
FEATURES_JSON = FEATURES_PATH / "features_dataset_hour_paths.json"

# ===== Globals for Features =====
feature_lookup = {}

def load_features():
    """Load and normalize features at startup."""
    global feature_lookup
    print("Loading feature vectors...")
    print(f"Looking for DAT file: {FEATURES_DAT}")
    print(f"Looking for JSON file: {FEATURES_JSON}")
    
    if not FEATURES_DAT.exists():
        print(f"❌ DAT file not found at: {FEATURES_DAT}")
        return
    if not FEATURES_JSON.exists():
        print(f"❌ JSON file not found at: {FEATURES_JSON}")
        return

    try:
        features = np.fromfile(str(FEATURES_DAT), dtype=np.float32)
        print(f"Raw features shape: {features.shape}")
        
        # Try to reshape - you might need to adjust the 2048 dimension
        # It could be different (e.g., 512, 1024, etc.)
        try:
            features = features.reshape(-1, 2048)
        except ValueError:
            # Try to detect the feature dimension
            total_elements = len(features)
            # Common feature dimensions: 512, 1024, 2048
            for dim in [512, 1024, 2048]:
                if total_elements % dim == 0:
                    features = features.reshape(-1, dim)
                    print(f"✅ Reshaped to dimension: {dim}")
                    break
            else:
                raise ValueError(f"Cannot reshape features of size {total_elements}")
        
        with open(FEATURES_JSON, 'r') as f:
            paths = json.load(f)
        
        print(f"Loaded {len(paths)} paths")

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        features = features / norms

        for i, path in enumerate(paths):
            filename = os.path.basename(path)
            feature_lookup[filename] = features[i]
        
        print(f"✅ Loaded {len(feature_lookup)} feature vectors.")
        print(f"Sample filenames: {list(feature_lookup.keys())[:5]}")
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        import traceback
        traceback.print_exc()

# ===== HTML (Shorter, Cleaner) =====
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>ReID Labeling</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family: Arial; background:#f0f2f5; height:100vh; display:flex; padding:10px; gap:10px; }
        
        /* Sidebar */
        .sidebar { 
            width:280px; background:white; border-radius:10px; padding:12px;
            display:flex; flex-direction:column; box-shadow:0 2px 8px rgba(0,0,0,0.1);
            flex-shrink:0;
        }
        .sidebar h3 { margin-bottom:8px; font-size:16px; display:flex; justify-content:space-between; }
        .sidebar h3 span { color:#666; font-weight:normal; font-size:13px; }
        
        .search { width:100%; padding:6px 10px; border:1px solid #ddd; border-radius:6px; margin-bottom:8px; }
        .search:focus { outline:none; border-color:#007bff; }
        
        .stats { display:flex; justify-content:space-between; font-size:12px; color:#666; background:#f8f9fa; padding:6px 10px; border-radius:6px; margin-bottom:8px; }
        
        .btn-group { display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }
        .btn { padding:8px 16px; border:none; border-radius:6px; cursor:pointer; font-size:14px; font-weight:bold; transition:0.2s; }
        .btn:hover { opacity:0.8; transform:translateY(-2px); }
        .btn-green { background:#28a745; color:white; }
        .btn-blue { background:#007bff; color:white; }
        .btn-red { background:#dc3545; color:white; }
        .btn-orange { background:#ffc107; color:black; }
        .btn-gray { background:#6c757d; color:white; }
        .btn-purple { background:#6f42c1; color:white; }
        
        .cluster-list { flex:1; overflow-y:auto; }
        .cluster-item { 
            padding:5px 10px; margin:2px 0; border-radius:4px; cursor:pointer;
            display:flex; justify-content:space-between; align-items:center;
            font-size:13px; transition:0.2s; border:2px solid transparent;
        }
        .cluster-item.drag-over { border: 2px dashed #007bff; background: #e7f3ff; }
        .cluster-item:hover { background:#e9ecef; }
        .cluster-item.active { background:#007bff; color:white; }
        .cluster-item .count { background:#ddd; color:#333; padding:0 8px; border-radius:10px; font-size:11px; }
        .cluster-item.active .count { background:white; color:#007bff; }
        .cluster-preview { width:80px; height:80px; border-radius:4px; object-fit:contain; margin-right:8px; background-color: #eee; }

        #similar-panel .cluster-item { flex-direction: column; align-items: flex-start; height: auto; padding: 10px; transition: all 0.2s; }
        #similar-panel .cluster-preview { width: 100%; height: 120px; margin-bottom: 8px; }
        #similar-panel .cluster-info { font-size: 11px; color: #666; }
        #similar-panel h4 { margin: 0; font-size: 14px; }
        #similar-panel .cluster-item.drag-over-similar { border: 2px solid #28a745; background: #f0fff4; box-shadow: 0 2px 8px rgba(40,167,69,0.2); }
        
        /* Gallery */
        .gallery {
            flex:1; background:white; border-radius:10px; padding:15px;
            display:flex; flex-direction:column; box-shadow:0 2px 8px rgba(0,0,0,0.1);
        }
        .gallery-header {
            display:flex; justify-content:space-between; align-items:center;
            padding-bottom:10px; border-bottom:1px solid #eee; flex-shrink:0;
        }
        .gallery-header h2 { font-size:18px; }
        .gallery-header .badge { background:#e9ecef; padding:2px 12px; border-radius:12px; font-size:13px; }
        
        .status { padding:8px 12px; border-radius:6px; margin-bottom:8px; display:none; font-size:13px; flex-shrink:0; }
        .status.show { display:block; }
        .status.success { background:#d4edda; color:#155724; }
        .status.error { background:#f8d7da; color:#721c24; }
        .status.info { background:#d1ecf1; color:#0c5460; }
        
        .image-grid {
            flex:1; overflow-y:auto; display:grid; 
            grid-template-columns:repeat(auto-fill, minmax(120px, 1fr));
            gap:8px; padding-top:10px; align-content:start;
        }
        .image-card {
            border:2px solid #ddd; border-radius:6px; overflow:hidden;
            cursor:grab; transition:0.2s; background:white; position:relative;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 150px;
        }
        .image-card:hover { transform:scale(1.03); border-color:#007bff; box-shadow:0 2px 8px rgba(0,0,0,0.15); }
        .image-card.selected { border-color:#28a745; background:#f0fff4; }
        .image-card.dragging { opacity:0.4; }
        .image-card img {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: contain;
            background:#f8f9fa;
        }
        .image-card .name { 
            font-size:9px; padding:3px 6px; text-align:center; 
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
            background:#f8f9fa; border-top:1px solid #eee;
        }
        .image-card .check {
            position:absolute; top:3px; right:3px; background:rgba(0,0,0,0.6);
            color:white; border-radius:50%; width:18px; height:18px;
            display:flex; align-items:center; justify-content:center;
            font-size:10px; opacity:0; transition:0.2s;
        }
        .image-card.selected .check { opacity:1; background:#28a745; }
        .image-card:hover .check { opacity:0.8; }
        
        .empty { color:#999; text-align:center; padding:40px; grid-column:1/-1; }
        .empty .icon { font-size:40px; display:block; margin-bottom:10px; }
        
        .controls { display:flex; gap:5px; }
        .controls button { padding:3px 10px; font-size:12px; border:1px solid #ddd; border-radius:4px; cursor:pointer; background:white; }
        .controls button:hover { background:#f0f0f0; }
        
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-thumb { background:#ccc; border-radius:2px; }
    </style>
</head>
<body>

<div class="sidebar">
    <h3>📁 Clusters <span id="totalImages">0</span></h3>
    <input class="search" id="search" placeholder="🔍 Search..." oninput="renderClusters()">
    <div class="stats">
        <span id="clusterCount">0 clusters</span>
        <span id="imageCount">0 images</span>
    </div>
    <div class="btn-group">
        <button class="btn btn-blue" onclick="newCluster()">✨ New</button>
        <button class="btn btn-red" onclick="splitCluster()">✂️ Split</button>
        <button class="btn btn-purple" onclick="findSimilar()">🔍 Find Similar</button>
        <button class="btn btn-green" onclick="exportLabels()">💾 Export</button>
        <button class="btn btn-gray" onclick="refresh()">🔄</button>
    </div>
    <div class="cluster-list" id="clusterList"></div>
</div>

<div class="gallery">
    <div class="gallery-header">
        <h2 id="clusterTitle">📂 Select a person</h2>
        <span class="badge" id="imageBadge">0 images</span>
    </div>
    
    <div class="status" id="status"></div>
    
    <div class="image-grid" id="imageGrid">
        <div class="empty"><span class="icon">👈</span>Select a person from the sidebar</div>
    </div>
</div>

<div class="sidebar" id="similar-panel" style="width: 250px;">
    <h3>🎯 Similar Clusters</h3>
    <div id="similar-results" class="cluster-list">
        <div class="empty" style="padding-top: 20px;">Select images and click "Find Similar"</div>
    </div>
</div>

<script>
let currentCluster = null;
let selected = new Set();
let clusterData = {};
let allImages = [];
let zoom = 120;
let draggedFile = null;
let dragSource = null;

// Load clusters
async function loadClusters() {
    try {
        const r = await fetch('/api/clusters');
        clusterData = await r.json();
        renderClusters();
        if (currentCluster && !clusterData[currentCluster]) {
            const keys = Object.keys(clusterData);
            currentCluster = keys.length ? keys[0] : null;
        }
        if (currentCluster) {
            selectCluster(currentCluster);
        } else if (Object.keys(clusterData).length > 0) {
            selectCluster(Object.keys(clusterData).sort((a,b) => parseInt(a.replace('person_','')) - parseInt(b.replace('person_','')))[0]);
        }
        else {
            document.getElementById('imageGrid').innerHTML = '<div class="empty"><span class="icon">👈</span>Select a person from the sidebar</div>';
            document.getElementById('clusterTitle').textContent = '📂 Select a person';
            document.getElementById('imageBadge').textContent = '0 images';
        }
        updateStats();
    } catch(e) { console.error(e); }
}

function renderClusters() {
    const search = document.getElementById('search').value.toLowerCase();
    const list = document.getElementById('clusterList');
    const sorted = Object.keys(clusterData).sort((a,b) => parseInt(a.replace('person_','')) - parseInt(b.replace('person_','')));
    
    list.innerHTML = '';
    let count = 0, total = 0;
    for (const key of sorted) {
        if (search && !key.toLowerCase().includes(search)) continue;
        count++;
        total += clusterData[key];
        const li = document.createElement('div');
        li.className = `cluster-item${currentCluster===key?' active':''}`;
        li.dataset.clusterId = key;
        li.innerHTML = `
            <span style="display:flex;align-items:center;flex:1;">
                <img class="cluster-preview" id="preview_${key}" src="/preview/${key}" 
                     onerror="this.style.display='none'">
                ${key}
            </span>
            <span class="count">${clusterData[key]}</span>
        `;
        li.addEventListener('click', () => selectCluster(key));
        li.addEventListener('dragover', e => { e.preventDefault(); li.classList.add('drag-over'); });
        li.addEventListener('dragleave', () => li.classList.remove('drag-over'));
        li.addEventListener('drop', onDropOnCluster);
        list.appendChild(li);
    }
    document.getElementById('clusterCount').textContent = `${count} clusters`;
    document.getElementById('imageCount').textContent = `${total} images`;
    document.getElementById('totalImages').textContent = `(${total})`;
}

function updateStats() {
    const total = Object.values(clusterData).reduce((a,b)=>a+b,0);
    document.getElementById('totalImages').textContent = `(${total})`;
}

async function selectCluster(name) {
    if (!name) return;
    currentCluster = name;
    selected.clear();
    document.querySelectorAll('.cluster-item').forEach(el => {
        el.classList.toggle('active', el.dataset.clusterId === name);
    });
    document.getElementById('clusterTitle').textContent = `📂 ${name}`;
    
    try {
        const r = await fetch(`/api/cluster/${name}`);
        const data = await r.json();
        allImages = data.images || [];
        document.getElementById('imageBadge').textContent = `${allImages.length} images`;
        renderImages();
    } catch(e) { console.error(e); }
}

function renderImages() {
    const grid = document.getElementById('imageGrid');
    if (!allImages.length) {
        grid.innerHTML = '<div class="empty"><span class="icon">📭</span>No images in this cluster</div>';
        return;
    }
    grid.innerHTML = allImages.map(img => `
        <div class="image-card ${selected.has(img)?'selected':''}" 
             draggable="true"
             onclick="toggle('${img}')"
             ondragstart="onDragStart(event,'${img}')"
             ondragend="onDragEnd(event)">
            <img src="/images/${currentCluster}/${img}" loading="lazy" alt="${img}" 
                 onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22%3E%3Crect fill=%22%23ddd%22 width=%22100%22 height=%22100%22/%3E%3Ctext x=%2250%22 y=%2250%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%23999%22 font-size=%2210%22%3E❌%3C/text%3E%3C/svg%3E'">
            <div class="name">${img}</div>
            <div class="check">✓</div>
        </div>
    `).join('');
    grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${zoom}px, 1fr))`;
}

function toggle(img) {
    selected.has(img) ? selected.delete(img) : selected.add(img);
    renderImages();
}

function selectAll() { allImages.forEach(i => selected.add(i)); renderImages(); }
function deselectAll() { selected.clear(); renderImages(); }
function zoomIn() { zoom = Math.min(zoom+20, 300); renderImages(); }
function zoomOut() { zoom = Math.max(zoom-20, 70); renderImages(); }

// Drag & Drop
function onDragStart(e, img) {
    draggedFile = img;
    dragSource = currentCluster;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', img);
    setTimeout(() => e.target.closest('.image-card')?.classList.add('dragging'), 0);
}
function onDragEnd(e) {
    e.target.closest('.image-card')?.classList.remove('dragging');
}

async function onDropOnCluster(e) {
    e.preventDefault();
    const targetCluster = e.currentTarget.dataset.clusterId;
    e.currentTarget.classList.remove('drag-over');
    
    if (!targetCluster) { showStatus('Invalid drop target', 'error'); return; }
    if (dragSource === targetCluster) { showStatus(`Already in ${targetCluster}`, 'info'); return; }
    
    // Move ALL selected images if any are selected, otherwise move the dragged one
    let imagesToMove = [];
    if (selected.size > 0 && dragSource === currentCluster) {
        imagesToMove = Array.from(selected);
    } else if (draggedFile) {
        imagesToMove = [draggedFile];
    } else {
        showStatus('No images to move', 'error');
        return;
    }
    
    await moveMultipleImages(imagesToMove, dragSource, targetCluster);
    draggedFile = null; dragSource = null;
}

// Similar panel drag & drop handlers
function onSimilarDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over-similar');
}

function onSimilarDragLeave(e) {
    e.currentTarget.classList.remove('drag-over-similar');
}

async function onDropOnSimilar(e, targetCluster) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over-similar');
    
    if (!targetCluster) {
        showStatus('Invalid drop target', 'error');
        return;
    }
    
    if (currentCluster === targetCluster) {
        showStatus(`Already in ${targetCluster}`, 'info');
        return;
    }
    
    // Check if we have selected images
    if (selected.size === 0) {
        showStatus('Select images first before dragging', 'error');
        return;
    }
    
    const imagesToMove = Array.from(selected);
    
    // Show count before moving
    showStatus(`📦 Moving ${imagesToMove.length} images to ${targetCluster}...`, 'info');
    
    await moveMultipleImages(imagesToMove, currentCluster, targetCluster);
}

async function moveImage(img, src, target) {
    try {
        const r = await fetch('/api/move', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({filename:img, source_cluster:src, target_cluster:target})
        });
        const result = await r.json();
        if (result.success) {
            showStatus(`✅ Moved ${img} to ${target}`, 'success');
            if (src === currentCluster) {
                allImages = allImages.filter(i => i !== img);
                selected.delete(img);
                renderImages();
            }
            clusterData[src]--;
            if(clusterData[target]){
                clusterData[target]++;
            } else {
                clusterData[target] = 1;
            }
            if (clusterData[src] === 0) {
                delete clusterData[src];

                if (currentCluster === src) {

                    const keys = Object.keys(clusterData).sort(
                        (a, b) =>
                            parseInt(a.replace('person_', '')) -
                            parseInt(b.replace('person_', ''))
                    );

                    const oldIndex = keys.findIndex(k =>
                        parseInt(k.replace('person_', '')) >
                        parseInt(src.replace('person_', ''))
                    );

                    currentCluster = oldIndex !== -1
                        ? keys[oldIndex]      // next folder
                        : (keys.length ? keys[keys.length - 1] : null); // last folder if no next exists

                    if (currentCluster)
                        selectCluster(currentCluster);
                }
            }

            renderClusters();
            updateStats();
        } else {
            showStatus(`❌ ${result.error}`, 'error');
        }
    } catch(e) { showStatus(`❌ ${e.message}`, 'error'); }
}

async function moveMultipleImages(images, src, target) {
    try {
        const r = await fetch('/api/move_multiple', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({filenames: images, source_cluster: src, target_cluster: target})
        });
        const result = await r.json();
        if (result.success) {
            showStatus(`✅ Moved ${images.length} images to ${target}`, 'success');
            if (src === currentCluster) {
                allImages = allImages.filter(i => !images.includes(i));
                images.forEach(img => selected.delete(img));
                renderImages();
            }
            clusterData[src] -= images.length;
            if(clusterData[target]){
                clusterData[target] += images.length;
            } else {
                clusterData[target] = images.length;
            }
            if (clusterData[src] === 0) {
                delete clusterData[src];

                if (currentCluster === src) {

                    const keys = Object.keys(clusterData).sort(
                        (a, b) =>
                            parseInt(a.replace('person_', '')) -
                            parseInt(b.replace('person_', ''))
                    );

                    const currentNumber = parseInt(src.replace('person_', ''));

                    let nextCluster = keys.find(k =>
                        parseInt(k.replace('person_', '')) > currentNumber
                    );

                    if (!nextCluster && keys.length > 0) {
                        nextCluster = keys[keys.length - 1];
                    }

                    currentCluster = nextCluster || null;

                    if (currentCluster) {
                        renderClusters();
                        await selectCluster(currentCluster);
                    } else {
                        renderClusters();
                    }

                    updateStats();
                    return;
                }
            }
            renderClusters();
            updateStats();
        } else {
            showStatus(`❌ ${result.error}`, 'error');
        }
    } catch(e) { showStatus(`❌ ${e.message}`, 'error'); }
}

function showStatus(msg, type='success') {
    const s = document.getElementById('status');
    s.textContent = msg;
    s.className = `status show ${type}`;
    clearTimeout(s._timer);
    s._timer = setTimeout(() => s.className = 'status', 3000);
}

// Export
async function exportLabels() {
    showStatus('📦 Exporting labels...', 'info');
    const r = await fetch('/api/export');
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `labels_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    showStatus('✅ Exported!', 'success');
}

// New Cluster
async function newCluster() {
    const newId = prompt("Enter new person ID (e.g., person_9999):");
    if (!newId || !/^person_\\d+$/.test(newId)) {

        showStatus("Invalid ID format. Use 'person_xxxx'.", 'error');
        return;
    }
    if (clusterData[newId]) {
        showStatus(`${newId} already exists.`, 'error');
        return;
    }

    const r = await fetch('/api/new_cluster', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cluster_id: newId})
    });
    const result = await r.json();
    if (result.success) {
        showStatus(`✅ Created new cluster: ${newId}`, 'success');
        clusterData[newId] = 0;
        renderClusters();
        selectCluster(newId);
    } else {
        showStatus(`❌ ${result.error}`, 'error');
    }
}

// Split
// Split
// Split
async function splitCluster() {
    if (!currentCluster) { showStatus('Select a cluster first', 'error'); return; }
    if (!selected.size) { showStatus('Select images to split first', 'error'); return; }
    
    // Find the next available cluster name
    function getNextAvailableClusterName() {
        // Get all existing cluster numbers
        const existingNumbers = Object.keys(clusterData)
            .filter(name => name.startsWith('person_'))
            .map(name => parseInt(name.replace('person_', '')))
            .sort((a, b) => a - b);
        
        // Find the smallest missing number starting from 1
        let nextNumber = 1;
        for (const num of existingNumbers) {
            if (num === nextNumber) {
                nextNumber++;
            } else if (num > nextNumber) {
                break;
            }
        }
        
        // Format with leading zeros (4 digits)
        return `person_${String(nextNumber).padStart(4, '0')}`;
    }
    
    const suggestedName = getNextAvailableClusterName();
    const newClusterId = prompt(
        `Enter new person ID for the ${selected.size} selected images:`,
        suggestedName  // Pre-fill with the suggested name
    );
    
    if (!newClusterId) return;
    
    // Validate format
    if (!/^person_\d+$/.test(newClusterId)) {
        showStatus("Invalid ID format. Use 'person_xxxx'.", 'error');
        return;
    }
    
    if (clusterData[newClusterId]) { 
        showStatus(`${newClusterId} already exists`, 'error'); 
        return; 
    }
    
    // Store the source cluster before moving
    const sourceCluster = currentCluster;
    const selectedImages = Array.from(selected);
    
    const r = await fetch('/api/split', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
            source_cluster:sourceCluster,
            target_cluster:newClusterId,
            images:selectedImages
        })
    });
    const result = await r.json();
    if (result.success) {
        showStatus(`✅ Created ${newClusterId} with ${selectedImages.length} images`, 'success');
        
        // Update cluster data
        clusterData[sourceCluster] -= selectedImages.length;
        clusterData[newClusterId] = selectedImages.length;
        
        // Remove source cluster if empty
        if (clusterData[sourceCluster] === 0) {
            delete clusterData[sourceCluster];
        }

        // Clear selection
        selected.clear();
        
        // Stay on the source cluster
        // If the source cluster still has images, stay on it
        if (clusterData[sourceCluster] > 0) {
            // Re-select the source cluster to refresh the view
            await selectCluster(sourceCluster);
        } else {
            // If source cluster is empty, find the next available cluster
            const keys = Object.keys(clusterData).sort(
                (a, b) => parseInt(a.replace('person_', '')) - parseInt(b.replace('person_', ''))
            );
            
            const currentNumber = parseInt(sourceCluster.replace('person_', ''));
            let nextCluster = keys.find(k => parseInt(k.replace('person_', '')) > currentNumber);
            
            if (!nextCluster && keys.length > 0) {
                nextCluster = keys[keys.length - 1];
            }
            
            if (nextCluster) {
                await selectCluster(nextCluster);
            } else {
                // No clusters left
                document.getElementById('imageGrid').innerHTML = '<div class="empty"><span class="icon">👈</span>Select a person from the sidebar</div>';
                document.getElementById('clusterTitle').textContent = '📂 Select a person';
                document.getElementById('imageBadge').textContent = '0 images';
                currentCluster = null;
            }
        }
        
        renderClusters();
        updateStats();

    } else {
        showStatus(`❌ ${result.error}`, 'error');
    }
}

async function refresh() {
    showStatus('🔄 Refreshing...', 'info');
    await loadClusters();
    showStatus('✅ Refreshed', 'success');
}

async function findSimilar() {
    if (!currentCluster || selected.size === 0) {
        showStatus('Select one or more images first.', 'info');
        return;
    }
    showStatus(`🔍 Searching for similar clusters...`, 'info');
    const resultsPanel = document.getElementById('similar-results');
    resultsPanel.innerHTML = '<div class="empty" style="padding-top: 20px;">Searching...</div>';

    try {
        const response = await fetch('/api/find_similar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cluster: currentCluster,
                images: Array.from(selected)
            })
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Server error');
        }
        const results = await response.json();
        renderSimilarClusters(results);
        showStatus(`✅ Found ${results.length} similar clusters. Drop selected images onto any cluster on the right to move them.`, 'success');
        
        // Scroll to show the similar panel
        document.getElementById('similar-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (e) {
        showStatus(`❌ Error: ${e.message}`, 'error');
        resultsPanel.innerHTML = '<div class="empty" style="padding-top: 20px;">Search failed.</div>';
    }
}

function renderSimilarClusters(results) {
    const panel = document.getElementById('similar-results');
    if (!results || results.length === 0) {
        panel.innerHTML = '<div class="empty" style="padding-top: 20px;">No similar clusters found.</div>';
        return;
    }
    panel.innerHTML = results.map(res => `
        <div class="cluster-item" 
             onclick="selectCluster('${res.cluster}')"
             ondragover="onSimilarDragOver(event)"
             ondragleave="onSimilarDragLeave(event)"
             ondrop="onDropOnSimilar(event, '${res.cluster}')"
             style="cursor:pointer;">
            <h4>${res.cluster}</h4>
            <img class="cluster-preview" src="/preview/${res.cluster}" onerror="this.style.display='none'">
            <div class="cluster-info">
                <strong>Similarity: ${res.score.toFixed(3)}</strong><br>
                ${res.count} images
                <br><small style="color:#28a745; font-weight:bold;">⬇️ Drop selected images here</small>
            </div>
        </div>
    `).join('');
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
    if (document.activeElement.tagName === 'INPUT') return;
    if (e.ctrlKey && e.key === 'a') { e.preventDefault(); selectAll(); }
    if (e.key === 'Escape') deselectAll();
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); exportLabels(); }
});

loadClusters();
</script>
</body>
</html>
'''

# ===== FLASK ROUTES =====

def get_cluster_images(cluster_dir):
    """Helper to get sorted image names from a directory."""
    if not cluster_dir.is_dir():
        return []
    return sorted([f.name for f in cluster_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/images/<cluster>/<filename>')
def serve_image(cluster, filename):
    return send_from_directory(str(CLUSTER_PATH / cluster), filename)

@app.route('/preview/<cluster>')
def preview_image(cluster):
    """Return first image from cluster for preview"""
    images = get_cluster_images(CLUSTER_PATH / cluster)
    if images:
        return send_from_directory(str(CLUSTER_PATH / cluster), images[0])
    return '', 404

@app.route('/api/clusters')
def get_clusters():
    clusters = {}
    if not CLUSTER_PATH.exists():
        return jsonify(clusters)
    for d in CLUSTER_PATH.iterdir():
        if d.is_dir() and d.name.startswith('person_'):
            count = len(get_cluster_images(d))
            clusters[d.name] = count
    return jsonify(clusters)

@app.route('/api/cluster/<name>')
def get_cluster(name):
    images = get_cluster_images(CLUSTER_PATH / name)
    return jsonify({'images': images})

@app.route('/api/new_cluster', methods=['POST'])
def new_cluster():
    data = request.json
    cluster_id = data.get('cluster_id')
    if not cluster_id:
        return jsonify({'success': False, 'error': 'Cluster ID is required'}), 400
    
    target_dir = CLUSTER_PATH / cluster_id
    if target_dir.exists():
        return jsonify({'success': False, 'error': 'Cluster already exists'}), 400
        
    target_dir.mkdir(exist_ok=True)
    return jsonify({'success': True})

@app.route('/api/move', methods=['POST'])
def move():
    data = request.json
    src, target, filename = data.get('source_cluster'), data.get('target_cluster'), data.get('filename')
    
    if not all([src, target, filename]):
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    src_path = CLUSTER_PATH / src / filename
    target_dir = CLUSTER_PATH / target
    target_path = target_dir / filename
    
    if not src_path.exists():
        return jsonify({'success': False, 'error': f'Source file not found: {src_path}'}), 404
    
    target_dir.mkdir(exist_ok=True)
    if target_path.exists():
        base, ext = os.path.splitext(filename)
        target_path = target_dir / f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
    
    shutil.move(str(src_path), str(target_path))
    
    # Clean up empty source directory if it's not the target
    src_dir = CLUSTER_PATH / src
    if src != target and not any(src_dir.iterdir()):
        try:
            src_dir.rmdir()
        except OSError as e:
            print(f"Error removing directory {src_dir}: {e}")

    return jsonify({'success': True, 'target': target})

@app.route('/api/merge', methods=['POST'])
def merge():
    data = request.json
    clusters, target = data.get('clusters'), data.get('target')
    
    if not all([clusters, target]) or len(clusters) < 1:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    target_dir = CLUSTER_PATH / target
    target_dir.mkdir(exist_ok=True)
    
    for src_name in set(clusters): # Use set to avoid merging a cluster with itself if duplicated
        if src_name == target:
            continue
        src_dir = CLUSTER_PATH / src_name
        if src_dir.exists():
            for img in src_dir.iterdir():
                if img.is_file():
                    dst = target_dir / img.name
                    if dst.exists():
                        base, ext = os.path.splitext(img.name)
                        dst = target_dir / f"{base}_m_{datetime.now().strftime('%H%M%S')}{ext}"
                    shutil.move(str(img), str(dst))
            # Clean up empty source directory
            if not any(src_dir.iterdir()):
                try:
                    src_dir.rmdir()
                except OSError as e:
                    print(f"Error removing directory {src_dir}: {e}")

    return jsonify({'success': True})

@app.route('/api/split', methods=['POST'])
def split():
    data = request.json
    src, target, images = data.get('source_cluster'), data.get('target_cluster'), data.get('images')

    if not all([src, target, images]):
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    target_dir = CLUSTER_PATH / target
    target_dir.mkdir(exist_ok=True)
    
    for img_name in images:
        src_file = CLUSTER_PATH / src / img_name
        if src_file.exists():
            dst = target_dir / img_name
            if dst.exists():
                base, ext = os.path.splitext(img_name)
                dst = target_dir / f"{base}_s_{datetime.now().strftime('%H%M%S')}{ext}"
            shutil.move(str(src_file), str(dst))

    # Clean up empty source directory if all images were moved
    src_dir = CLUSTER_PATH / src
    if not any(src_dir.iterdir()):
        try:
            src_dir.rmdir()
        except OSError as e:
            print(f"Error removing directory {src_dir}: {e}")

    return jsonify({'success': True})

@app.route('/api/export')
def export():
    labels = {}
    if not CLUSTER_PATH.exists():
        return jsonify(labels);
        
    for d in CLUSTER_PATH.iterdir():
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
    if not feature_lookup:
        return jsonify({'success': False, 'error': 'Feature lookup is not available.'}), 500

    data = request.json
    source_cluster = data.get('cluster')
    image_names = data.get('images')

    if not all([source_cluster, image_names]):
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    # 1. Obtain feature vectors for selected images
    query_vectors = []
    for img_name in image_names:
        if img_name in feature_lookup:
            query_vectors.append(feature_lookup[img_name])
    
    if not query_vectors:
        return jsonify({'success': False, 'error': 'None of the selected images have feature vectors.'}), 404

    # 2. Compute the average feature and normalize it
    query_feature = np.mean(query_vectors, axis=0)
    query_feature /= np.linalg.norm(query_feature)

    # 3. Iterate through all other clusters
    similarities = []
    for cluster_dir in CLUSTER_PATH.iterdir():
        if not cluster_dir.is_dir() or not cluster_dir.name.startswith('person_'):
            continue
        
        # 4. Skip the current folder
        if cluster_dir.name == source_cluster:
            continue

        # 5. For every image, compute similarity and find the max
        max_sim = -1.0
        image_files = get_cluster_images(cluster_dir)
        if not image_files:
            continue

        for img_name in image_files:
            if img_name in feature_lookup:
                img_feature = feature_lookup[img_name]
                # 6. Cosine similarity
                similarity = np.dot(query_feature, img_feature)
                if similarity > max_sim:
                    max_sim = similarity

        if max_sim > -1.0:
            similarities.append({
                "cluster": cluster_dir.name,
                "score": float(max_sim),
                "count": len(image_files),
                "preview": image_files[0] if image_files else None
            })

    # 7. Sort and return top 5
    similarities.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(similarities[:5])

@app.route('/api/move_multiple', methods=['POST'])
def move_multiple():
    data = request.json
    src, target, filenames = data.get('source_cluster'), data.get('target_cluster'), data.get('filenames')
    
    if not all([src, target, filenames]) or not isinstance(filenames, list):
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    target_dir = CLUSTER_PATH / target
    target_dir.mkdir(exist_ok=True)
    moved_count = 0
    
    for filename in filenames:
        src_path = CLUSTER_PATH / src / filename
        if src_path.exists():
            target_path = target_dir / filename
            if target_path.exists():
                base, ext = os.path.splitext(filename)
                target_path = target_dir / f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
            shutil.move(str(src_path), str(target_path))
            moved_count += 1

    # Clean up empty source directory
    src_dir = CLUSTER_PATH / src
    if src != target and not any(src_dir.iterdir()):
        try:
            src_dir.rmdir()
        except OSError as e:
            print(f"Error removing directory {src_dir}: {e}")

    return jsonify({'success': True, 'moved': moved_count, 'target': target})

if __name__ == '__main__':
    load_features()
    print(f"\n🚀 ReID Labeling Tool")
    if not CLUSTER_PATH.exists():
        print(f"⚠️  Warning: Cluster path not found. Creating directory...")
        CLUSTER_PATH.mkdir(parents=True, exist_ok=True)
    print(f"📂 Serving from: {CLUSTER_PATH}")
    print(f"🌐 http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)