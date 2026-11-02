"""Step 1: run MediaPipe over every dataset image and save hand landmarks.

Usage:
    python extract_landmarks.py                       # downloads the Kaggle dataset via kagglehub
    python extract_landmarks.py --data /path/to/ds    # use a folder you already have
    python extract_landmarks.py --max-per-class 300   # quick test run

The dataset is expected to be organised as   <something>/<CLASS_NAME>/<image files>
(any depth). The class is taken from the image's parent folder name.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import LANDMARKS_PATH, N_LM, create_landmarker, detect_hands

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
KAGGLE_ID = "rushilverma07/indian-sign-language-alphabet-dataset"


def get_dataset_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_ID)
    print("Dataset at:", path)
    return Path(path)


def collect_images(root: Path) -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
        if imgs:
            label = Path(dirpath).name.upper()
            by_label[label] += [os.path.join(dirpath, f) for f in imgs]
    return by_label


def detect_with_padding(landmarker, bgr: np.ndarray):
    """Hand detectors dislike hands that touch the image border -> retry with padding."""
    h, w = bgr.shape[:2]
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
        h, w = bgr.shape[:2]
    hands = detect_hands(landmarker, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if hands:
        return hands
    pad = int(0.3 * max(h, w))
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return detect_hands(landmarker, cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="dataset folder (default: download with kagglehub)")
    ap.add_argument("--out", default=str(LANDMARKS_PATH))
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = use all images")
    ap.add_argument("--include-digits", action="store_true",
                    help="also keep classes that are not a single letter (e.g. 1-9)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = get_dataset_root(args.data)
    by_label = collect_images(root)
    print(f"Found {len(by_label)} folders with images: {sorted(by_label)[:50]}")
    if len(by_label) > 45:
        print("WARNING: that is a lot of classes - the dataset layout may not be "
              "<class>/<images>. Check the folder structure.")

    if not args.include_digits:
        by_label = {k: v for k, v in by_label.items() if len(k) == 1 and k.isalpha()}
    if not by_label:
        raise SystemExit("No letter classes found. Check the folder names above, "
                         "or pass --include-digits.")

    rng = random.Random(args.seed)
    landmarker = create_landmarker(video=False, num_hands=2, min_det=0.3)

    raws, n_hands, labels = [], [], []
    stats = {}
    for label in sorted(by_label):
        paths = by_label[label]
        if args.max_per_class and len(paths) > args.max_per_class:
            paths = rng.sample(paths, args.max_per_class)
        ok = 0
        two = 0
        for p in tqdm(paths, desc=f"class {label}", leave=False):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            hands = detect_with_padding(landmarker, bgr)
            if not hands:
                continue
            slot = np.zeros((2, N_LM, 3), np.float32)
            for i, h in enumerate(hands[:2]):
                slot[i] = h
            raws.append(slot)
            n_hands.append(min(len(hands), 2))
            labels.append(label)
            ok += 1
            two += len(hands) >= 2
        stats[label] = (ok, len(paths), two)
        print(f"  {label}: {ok}/{len(paths)} detected ({100 * ok / max(len(paths), 1):.0f}%), "
              f"two-hand frames: {100 * two / max(ok, 1):.0f}%")

    if not raws:
        raise SystemExit("No hands detected in any image.")
    np.savez_compressed(args.out, raw=np.stack(raws), n_hands=np.array(n_hands),
                        labels=np.array(labels))
    low = [k for k, (ok, tot, _) in stats.items() if ok / max(tot, 1) < 0.5]
    print(f"\nSaved {len(raws)} samples -> {args.out}")
    if low:
        print("Classes with <50% detection rate (consider checking these images):", low)


if __name__ == "__main__":
    main()
