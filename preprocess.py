import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from extractor import MediaPipeExtractor

DATASET_DIR = "dataset"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "landmarks.csv")

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Directory '{DATASET_DIR}' not found. Please place your images there.")
        return
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    extractor = MediaPipeExtractor()
    
    # Define header: label, f0...f125 (126 features)
    header = ["label"] + [f"f{i}" for i in range(126)]
    rows = []
    
    # Get all subdirectories (A-Z, 0-9), ignoring hidden OS files
    classes = sorted([d for d in os.listdir(DATASET_DIR) 
                      if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')])
    print(f"Found {len(classes)} classes in dataset.")
    
    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        images = [img for img in os.listdir(cls_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\nProcessing class '{cls}'...")
        for img_name in tqdm(images):
            img_path = os.path.join(cls_dir, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                continue
                
            features, _ = extractor.extract(frame)
            
            # Skip frames where no hands are detected
            if np.all(features == 0.0):
                continue
                
            row = [cls] + features.tolist()
            rows.append(row)
            
    # Save the aggregated features
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} valid feature vectors to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
