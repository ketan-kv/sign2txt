import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from image_utils import get_canny_edge

DATASET_DIR = "dataset"
IMG_SIZE = 128

def load_data():
    X, y = [], []
    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} folder not found.")
        return np.array([]), np.array([]), 0

    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    label_map = {str(i): cls for i, cls in enumerate(classes)}
    with open("labels.json", "w") as f:
        json.dump(label_map, f)
        
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(DATASET_DIR, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Apply identical preprocessing as live inference
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            edges = get_canny_edge(img)
            
            # Normalize and format to (128, 128, 1)
            edges = edges.astype('float32') / 255.0
            edges = np.expand_dims(edges, axis=-1)
            
            X.append(edges)
            y.append(i)
            
    return np.array(X), np.array(y), len(classes)

def build_model(num_classes):
    # Literature-validated lightweight Tier 1 Architecture
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading and converting dataset into 1-channel Canny Edge maps...")
    X, y, num_classes = load_data()
    
    if len(X) == 0:
        print("No images found to train on. Exiting.")
        exit()
        
    print(f"Loaded {len(X)} images across {num_classes} classes.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(num_classes)
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    
    print("\nTraining Tier-1 Custom Edge-CNN...")
    model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=30, 
        batch_size=32, 
        callbacks=[early_stop]
    )
    
    model.save("isl_cnn_model.keras")
    print("\nTraining complete! Saved to isl_cnn_model.keras")
