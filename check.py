import json
from pathlib import Path

with open("C:/Users/Manar/Documents/REID/Cluster_id/extracted_features/2026041316/features_dataset_paths.json", 'r') as f:
    paths = json.load(f)
    print("First 5 filenames in features:")
    for p in paths[:5]:
        print(Path(p).name)