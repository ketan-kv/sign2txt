import cv2
import numpy as np

def get_canny_edge(frame_crop):
    """
    Converts a BGR image crop into a 2D Canny Edge map.
    This strips away skin tone, lighting, and background color,
    leaving only structural white lines.
    """
    gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Thresholds may need tuning based on webcam lighting
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def preprocess_for_cnn(frame_crop, img_size=128):
    """
    Runs the Canny edge detection, resizes, normalizes,
    and formats the image exactly as the CNN expects (H, W, 1).
    """
    resized = cv2.resize(frame_crop, (img_size, img_size))
    edges = get_canny_edge(resized)
    
    # Normalize pixel values to 0-1
    normalized = edges.astype('float32') / 255.0
    
    # Expand dims to add the channel dimension: (128, 128, 1)
    return np.expand_dims(normalized, axis=-1)
