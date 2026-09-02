import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def main():
    print("Loading landmark data...")
    try:
        df = pd.read_csv("data/landmarks.csv", low_memory=False)
    except FileNotFoundError:
        print("Error: data/landmarks.csv not found. Run preprocess.py first.")
        return
        
    if df.shape[1] != 133:
        raise ValueError(f"Expected 133 columns (1 label + 132 features), got {df.shape[1]}")
        
    X = df.iloc[:, 1:].values
    y_raw = df.iloc[:, 0].astype(str).values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    label_mapping = {int(index): str(label) for index, label in enumerate(le.classes_)}
    with open("labels.json", "w") as f:
        json.dump(label_mapping, f)
        
    num_classes = len(le.classes_)
    print(f"Loaded {len(X)} samples across {num_classes} classes.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Built to handle 132 features and robust against overfitting (Dropout 0.4)
    model = Sequential([
        Dense(256, activation='relu', input_shape=(132,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    print("\nStarting Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop]
    )
    
    model.save("isl_dnn_model.keras")
    print("Training complete! Model saved to isl_dnn_model.keras")

if __name__ == "__main__":
    main()
