import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipeExtractor:
    def __init__(self):
        task_path = "hand_landmarker.task"
        if not os.path.exists(task_path):
            print("Downloading MediaPipe model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 
                task_path
            )
            
        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2, 
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,4),       
            (0,5), (5,6), (6,7), (7,8),       
            (5,9), (9,10), (10,11), (11,12),  
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (0,17), (17,18), (18,19), (19,20) 
        ]
        
    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)
        
        # 132 Features: 2 hands * (21 landmarks * 3 coords + 3 palm normal vector)
        features = np.zeros(132, dtype=np.float32)
        
        if not results.hand_landmarks:
            return features, results
            
        # Sort hands by wrist X-coordinate
        hands_data = sorted(results.hand_landmarks, key=lambda hl: hl[0].x)
        
        # 1. Global Normalization: Find the midpoint of all detected wrists
        wrists = [hl[0] for hl in hands_data]
        anchor_x = sum([w.x for w in wrists]) / len(wrists)
        anchor_y = sum([w.y for w in wrists]) / len(wrists)
        anchor_z = sum([w.z for w in wrists]) / len(wrists)
        
        # 2. Scale Invariance: Find maximum distance from anchor to any point
        max_dist = 1e-6
        for hl in hands_data:
            for lm in hl:
                dist = np.sqrt((lm.x - anchor_x)**2 + (lm.y - anchor_y)**2 + (lm.z - anchor_z)**2)
                max_dist = max(max_dist, dist)
                
        for i, hl in enumerate(hands_data):
            if i >= 2: break
            
            hand_features = []
            
            # Append normalized scale-invariant coordinates (63 features)
            for lm in hl:
                hand_features.extend([
                    (lm.x - anchor_x) / max_dist,
                    (lm.y - anchor_y) / max_dist,
                    (lm.z - anchor_z) / max_dist
                ])
                
            # 3. Palm Orientation: Calculate Palm Normal Vector (3 features)
            v1_x, v1_y, v1_z = hl[5].x - hl[0].x, hl[5].y - hl[0].y, hl[5].z - hl[0].z
            v2_x, v2_y, v2_z = hl[17].x - hl[0].x, hl[17].y - hl[0].y, hl[17].z - hl[0].z
            
            # Cross Product
            nx = v1_y * v2_z - v1_z * v2_y
            ny = v1_z * v2_x - v1_x * v2_z
            nz = v1_x * v2_y - v1_y * v2_x
            
            # Normalize Vector
            n_mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
            hand_features.extend([nx/n_mag, ny/n_mag, nz/n_mag])
            
            # Insert into feature array
            start_idx = i * 66
            features[start_idx:start_idx+66] = hand_features
            
        return features, results
        
    def draw_landmarks(self, frame, results):
        if results and results.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.hand_landmarks:
                for p1, p2 in self.CONNECTIONS:
                    x1, y1 = int(hand_landmarks[p1].x * w), int(hand_landmarks[p1].y * h)
                    x2, y2 = int(hand_landmarks[p2].x * w), int(hand_landmarks[p2].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        return frame
