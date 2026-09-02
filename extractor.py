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
        
        # 126 Features: Left Hand (63) + Right Hand (63)
        features = np.zeros(126, dtype=np.float32)
        
        if not results.hand_landmarks:
            return features, results
            
        extracted_hands = []
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = results.handedness[i][0].category_name
            wrist = hand_landmarks[0]
            
            hand_features = []
            for lm in hand_landmarks:
                hand_features.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])
                
            extracted_hands.append({
                "handedness": handedness,
                "x_coord": wrist.x,
                "features": hand_features
            })
            
        if len(extracted_hands) == 1:
            # Single-Hand Fallback
            start_idx = 0 if extracted_hands[0]["handedness"] == "Left" else 63
            features[start_idx:start_idx+63] = extracted_hands[0]["features"]
            
        elif len(extracted_hands) >= 2:
            h1, h2 = extracted_hands[0], extracted_hands[1]
            
            # Check if MediaPipe correctly labeled one Left and one Right
            if h1["handedness"] != h2["handedness"]:
                idx1 = 0 if h1["handedness"] == "Left" else 63
                idx2 = 0 if h2["handedness"] == "Left" else 63
                features[idx1:idx1+63] = h1["features"]
                features[idx2:idx2+63] = h2["features"]
            else:
                # CONFLICT: MediaPipe got confused by occlusion (e.g. M, N, H)
                # and labeled both hands as 'Left' or both as 'Right'.
                # Fallback: The hand physically on the left side of the mirrored screen is the physical Left hand.
                left_hand = h1 if h1["x_coord"] < h2["x_coord"] else h2
                right_hand = h2 if h1["x_coord"] < h2["x_coord"] else h1
                
                features[0:63] = left_hand["features"]
                features[63:126] = right_hand["features"]
                
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
