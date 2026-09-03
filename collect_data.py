import cv2
import os
import time
import numpy as np
from image_utils import get_canny_edge

def main():
    print("======================================================")
    print(" ISL Domain Gap Data Collector (Edge-CNN Edition)")
    print("======================================================")
    
    class_name = input("Enter the Symbol you want to record (e.g., 'H', 'M', 'N'): ").strip().upper()
    if not class_name:
        print("Invalid class name. Exiting.")
        return
        
    save_dir = os.path.join("webcam_dataset", class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    WIDTH, HEIGHT = 640, 480
    BOX_X1 = int(WIDTH * 0.15)
    BOX_Y1 = int(HEIGHT * 0.15)
    BOX_X2 = int(WIDTH * 0.85)
    BOX_Y2 = int(HEIGHT * 0.85)
    
    count = 0
    print("\nCONTROLS: [SPACE] to capture, [Q] to quit.")
    print("IMPORTANT: Watch the 'Canny AI View' window! Make sure your hands show up clearly as white lines.")
    print("If your background is too messy, the AI will get confused. Try moving to a plainer wall.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        display_frame = frame.copy()
        
        # 1. Extract Crop
        hand_crop = frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2]
        
        # 2. Show what the AI will see (Canny Edge)
        ai_view = get_canny_edge(hand_crop)
        cv2.imshow("What the AI Sees (Canny Edge)", ai_view)
        
        # Draw UI
        cv2.rectangle(display_frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Class: {class_name} | Captured: {count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        cv2.imshow("Data Collector", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # Spacebar
            # We save the RAW color image, so we can tune Canny parameters later if needed,
            # but train_cnn.py will convert this to Canny before training.
            filename = os.path.join(save_dir, f"webcam_{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, hand_crop)
            count += 1
            print(f"Captured {count}: {filename}")
            
            flash = np.full(display_frame.shape, 255, dtype=np.uint8)
            cv2.imshow("Data Collector", flash)
            cv2.waitKey(50)
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Captured {count} images for class '{class_name}'.")

if __name__ == "__main__":
    main()
