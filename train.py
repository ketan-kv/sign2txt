"""Step 2: train a small classifier on the extracted landmarks.

Usage:  python train.py [--aug 4]
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import CLASSIFIER_PATH, FEATURE_DIM, LANDMARKS_PATH, augment_hands, hands_to_features


def build(raw, n_hands, idx, rng, n_aug):
    X, y_idx = [], []
    for i in idx:
        hands = [raw[i, j] for j in range(int(n_hands[i]))]
        X.append(hands_to_features(hands))
        y_idx.append(i)
        for _ in range(n_aug):
            X.append(hands_to_features(augment_hands(hands, rng)))
            y_idx.append(i)
    return np.stack(X), np.array(y_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(LANDMARKS_PATH))
    ap.add_argument("--aug", type=int, default=4, help="augmented copies per training sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data)
    raw, n_hands, labels = d["raw"], d["n_hands"], d["labels"]
    print(f"{len(labels)} samples, {len(set(labels))} classes")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(labels))
    tr, va = train_test_split(idx, test_size=0.15, stratify=labels, random_state=args.seed)

    Xtr, itr = build(raw, n_hands, tr, rng, args.aug)
    Xva, iva = build(raw, n_hands, va, rng, 0)       # validation: untouched originals
    # Integer-encode labels ourselves (sklearn's MLP early-stopping breaks on string labels)
    enc = LabelEncoder().fit(labels)
    ytr, yva = enc.transform(labels[itr]), enc.transform(labels[iva])
    assert Xtr.shape[1] == FEATURE_DIM

    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True,
                      n_iter_no_change=8, max_iter=200, random_state=args.seed, verbose=True),
    )
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xva)
    print(f"\nValidation accuracy: {(pred == yva).mean():.3f}")
    rep = classification_report(yva, pred, output_dict=True, zero_division=0)
    names = {str(i): c for i, c in enumerate(enc.classes_)}
    worst = sorted((v["recall"], names[k]) for k, v in rep.items() if k in names)[:6]
    print("Weakest classes (recall):", ", ".join(f"{k}={r:.2f}" for r, k in worst))
    print("NOTE: the dataset likely has many near-identical frames, so this number is "
          "optimistic. Real accuracy on YOUR webcam is what matters.")

    joblib.dump({"pipeline": pipe, "classes": [str(c) for c in enc.classes_]}, CLASSIFIER_PATH)
    print("Saved", CLASSIFIER_PATH)


if __name__ == "__main__":
    main()
