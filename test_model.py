"""
test_model.py — Per-Class Accuracy Audit Tool

Loads the trained DNN and the landmarks CSV, runs predictions on the
full test split, and prints a detailed confusion matrix + per-class
precision/recall/F1 report.

Usage:
    python test_model.py
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def main():
    print("Loading model and data...")
    
    try:
        df = pd.read_csv("data/landmarks.csv", low_memory=False)
    except FileNotFoundError:
        print("Error: data/landmarks.csv not found. Run preprocess.py first.")
        return

    model = tf.keras.models.load_model("isl_dnn_model.keras")

    with open("labels.json", "r") as f:
        label_map = json.load(f)

    X = df.iloc[:, 1:].values
    y_raw = df.iloc[:, 0].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Use the same split parameters as train.py so we evaluate on the exact same test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Run predictions
    print(f"Evaluating on {len(X_test)} test samples...\n")
    predictions = model(X_test, training=False).numpy()
    y_pred = np.argmax(predictions, axis=1)

    # Build human-readable class names in label order
    target_names = [label_map[str(i)] for i in range(len(le.classes_))]

    # Per-class report
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print("=" * 60)
    print("PER-CLASS ACCURACY REPORT")
    print("=" * 60)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("=" * 60)
    print("CONFUSION MATRIX (rows = actual, cols = predicted)")
    print("=" * 60)

    # Print header
    header = "     " + "  ".join(f"{n:>3}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>3}" for v in row)
        print(f"{target_names[i]:>3}  {row_str}")

    # Highlight worst-performing classes
    print("\n" + "=" * 60)
    print("SYMBOLS BELOW 85% RECALL (NEED ATTENTION)")
    print("=" * 60)

    per_class_recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    flagged = False
    for i, recall in enumerate(per_class_recall):
        if recall < 0.85:
            flagged = True
            # Find what it's most confused with
            row = cm[i].copy()
            row[i] = 0  # exclude self
            worst_idx = np.argmax(row)
            confused_with = target_names[worst_idx]
            confused_count = row[worst_idx]
            total = cm[i].sum()
            print(f"  [!] {target_names[i]:>3}: {recall*100:.1f}% recall -- "
                  f"most confused with '{confused_with}' ({confused_count}/{total} misclassified)")

    if not flagged:
        print("  [OK] All symbols are above 85% recall! Stage 1 gate passed.")


if __name__ == "__main__":
    main()
