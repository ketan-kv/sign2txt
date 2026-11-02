"""Shared helpers: MediaPipe hand-landmark model, feature engineering, augmentation.

Why landmarks instead of a CNN on raw pixels?
  * Robust to background, lighting, skin tone, clothing and camera distance.
  * The classifier is tiny and runs in <1 ms per frame (no GPU needed).
  * Training on a dataset recorded in one place still transfers to your webcam.

ISL fingerspelling uses BOTH hands for many letters, so features cover up to two hands
(including where the hands are relative to each other).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = HERE / "hand_landmarker.task"
CLASSIFIER_PATH = HERE / "isl_classifier.joblib"
LANDMARKS_PATH = HERE / "landmarks.npz"

N_LM = 21                                # landmarks per hand
FEATURE_DIM = 2 * N_LM * 3 + 2           # 2 hand slots x (21 x xyz) + 2 "present" flags

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --------------------------------------------------------------------------- MediaPipe
def ensure_hand_model(path: Path = HAND_MODEL_PATH) -> Path:
    """Download MediaPipe's hand_landmarker.task once (~7.5 MB)."""
    if not path.exists():
        print(f"Downloading hand landmark model to {path} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def create_landmarker(video: bool, num_hands: int = 2,
                      min_det: float = 0.5, min_track: float = 0.5):
    """video=True -> live webcam (tracks between frames); False -> independent still images."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    ensure_hand_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_det,
        min_hand_presence_confidence=min_det,
        min_tracking_confidence=min_track,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, rgb: np.ndarray, ts_ms: int | None = None) -> list[np.ndarray]:
    """Run the landmarker on an RGB uint8 image. Returns a list of (21, 3) arrays.

    Coordinates are in pixel-like units (x*W, y*H, z*W) so geometry isn't distorted by
    the image aspect ratio.
    """
    import mediapipe as mp

    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = (landmarker.detect_for_video(mp_image, ts_ms) if ts_ms is not None
              else landmarker.detect(mp_image))
    hands = []
    for lms in result.hand_landmarks:
        hands.append(np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], np.float32))
    return hands


# --------------------------------------------------------------------------- features
def _hand_scale(h: np.ndarray) -> float:
    """Palm size: robust to the hand being foreshortened in one direction."""
    return max(float(np.linalg.norm(h[9] - h[0])), float(np.linalg.norm(h[5] - h[17])), 1e-6)


def hands_to_features(hands: list[np.ndarray]) -> np.ndarray:
    """Convert 0-2 hands into a fixed-length, translation/scale-invariant vector.

    * Hands are ordered left-to-right on screen (NOT by handedness label, which flips
      when the image is mirrored).
    * Everything is expressed relative to the wrist (one hand) or the midpoint between
      wrists (two hands) and divided by palm size -> position/distance invariant, but the
      relative placement of the two hands is preserved.
    """
    feat = np.zeros(FEATURE_DIM, np.float32)
    if not hands:
        return feat
    hands = sorted(hands[:2], key=lambda h: float(h[0, 0]))
    origin = np.mean([h[0] for h in hands], axis=0)
    scale = float(np.mean([_hand_scale(h) for h in hands]))
    for i, h in enumerate(hands):
        feat[i * N_LM * 3:(i + 1) * N_LM * 3] = ((h - origin) / scale).ravel()
        feat[2 * N_LM * 3 + i] = 1.0
    return feat


def augment_hands(hands: list[np.ndarray], rng: np.random.Generator,
                  max_rot_deg: float = 15.0, noise: float = 0.015) -> list[np.ndarray]:
    """Random mirror + small in-plane rotation + landmark jitter."""
    flip = rng.random() < 0.5
    ang = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]], np.float32)
    center = np.mean([h[:, :2].mean(axis=0) for h in hands], axis=0)
    out = []
    for h in hands:
        h = h.copy()
        if flip:
            h[:, 0] = -h[:, 0]
            cx = -center[0]
        else:
            cx = center[0]
        c2 = np.array([cx, center[1]], np.float32)
        h[:, :2] = (h[:, :2] - c2) @ rot.T + c2
        h += rng.normal(0.0, noise * _hand_scale(h), h.shape).astype(np.float32)
        out.append(h)
    return out
