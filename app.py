import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json
import os
import time
from extractor import MediaPipeExtractor

class ISLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Detection - Snapshot Mode")
        self.root.geometry("800x800")
        self.root.configure(bg="#2b2b2b")
        
        if not os.path.exists("isl_dnn_model.keras") or not os.path.exists("labels.json"):
            print("Error: Required model files missing. Run train.py first.")
            self.root.destroy()
            return
            
        self.model = tf.keras.models.load_model("isl_dnn_model.keras")
        with open("labels.json", "r") as f:
            self.class_names = json.load(f)
            
        self.extractor = MediaPipeExtractor()
        self.cap = cv2.VideoCapture(0)
        
        # Snapshot Mode State
        self.snapshot_timer_start = None
        self.cooldown_until = 0.0
        self.flash_frames = 0
        
        # UI Constants
        self.WIDTH, self.HEIGHT = 640, 480
        # Target Box taking up ~70% of the screen center
        self.BOX_X1 = int(self.WIDTH * 0.15)
        self.BOX_Y1 = int(self.HEIGHT * 0.15)
        self.BOX_X2 = int(self.WIDTH * 0.85)
        self.BOX_Y2 = int(self.HEIGHT * 0.85)
        
        self.setup_ui()
        self.update_frame()
        
    def setup_ui(self):
        self.vid_label = tk.Label(self.root, bg="#2b2b2b")
        self.vid_label.pack(pady=10)
        
        self.text_box = tk.Text(self.root, height=3, width=40, font=("Helvetica", 24, "bold"), bg="#f0f0f0", fg="black")
        self.text_box.pack(pady=10, padx=20)
        
        btn_frame = tk.Frame(self.root, bg="#2b2b2b")
        btn_frame.pack(pady=10)
        
        self.btn_space = tk.Button(btn_frame, text="Add Space", font=("Helvetica", 12, "bold"), 
                                 bg="#4caf50", fg="white", command=self.add_space)
        self.btn_space.grid(row=0, column=0, padx=10, pady=5, ipadx=10, ipady=5)
        
        self.btn_backspace = tk.Button(btn_frame, text="Backspace", font=("Helvetica", 12, "bold"), 
                                     bg="#ff9800", fg="white", command=self.backspace)
        self.btn_backspace.grid(row=0, column=1, padx=10, pady=5, ipadx=10, ipady=5)
        
        self.btn_clear = tk.Button(btn_frame, text="Clear All", font=("Helvetica", 12, "bold"), 
                                 bg="#f44336", fg="white", command=self.clear_text)
        self.btn_clear.grid(row=0, column=2, padx=10, pady=5, ipadx=10, ipady=5)
        
        self.btn_save = tk.Button(btn_frame, text="Save to Text", font=("Helvetica", 12, "bold"), 
                                bg="#2196f3", fg="white", command=self.save_file)
        self.btn_save.grid(row=1, column=0, columnspan=1, padx=10, pady=10, ipadx=10, ipady=5)
        
        self.btn_quit = tk.Button(btn_frame, text="Quit", font=("Helvetica", 12, "bold"), 
                                bg="#555555", fg="white", command=self.quit_app)
        self.btn_quit.grid(row=1, column=2, columnspan=1, padx=10, pady=10, ipadx=10, ipady=5)

    def add_space(self): self.text_box.insert(tk.END, " ")
    def clear_text(self): self.text_box.delete("1.0", tk.END)
    
    def backspace(self):
        current = self.text_box.get("1.0", tk.END)
        if len(current) > 1:
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, current[:-2])
            
    def save_file(self):
        fp = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if fp:
            with open(fp, "w") as f: f.write(self.text_box.get("1.0", tk.END))
                
    def quit_app(self):
        self.cap.release()
        try: self.extractor.detector.close()
        except: pass
        self.root.destroy()
        
    def get_prediction(self, features):
        input_data = np.expand_dims(features, axis=0)
        predictions = self.model(input_data, training=False).numpy()
        max_idx = np.argmax(predictions[0])
        return self.class_names[str(max_idx)], predictions[0][max_idx]
        
    def trigger_snapshot(self, pred_char):
        if pred_char.lower() == "space":
            self.text_box.insert(tk.END, " ")
        elif pred_char.lower() in ["delete", "del", "backspace"]:
            self.backspace()
        else:
            self.text_box.insert(tk.END, pred_char)
            
        # Trigger screen flash and cooldown
        self.flash_frames = 3
        self.cooldown_until = time.time() + 1.5
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            
            features, results = self.extractor.extract(frame)
            frame = self.extractor.draw_landmarks(frame, results)
            
            # Check if hand is inside the large target box
            in_box = False
            if results and results.hand_landmarks:
                in_box = True
                for hand_landmarks in results.hand_landmarks:
                    wrist = hand_landmarks[0]
                    wx, wy = int(wrist.x * self.WIDTH), int(wrist.y * self.HEIGHT)
                    if not (self.BOX_X1 <= wx <= self.BOX_X2 and self.BOX_Y1 <= wy <= self.BOX_Y2):
                        in_box = False
                        break
                        
            # Live Debug Info
            display_text = "Prediction: None"
            color = (0, 0, 255)
            if not np.all(features == 0.0):
                pred_char, conf = self.get_prediction(features)
                display_text = f"Prediction: {pred_char} ({conf*100:.1f}%)"
                color = (0, 255, 0) if conf > 0.6 else (0, 165, 255)
                
            cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw Target Box
            box_color = (0, 255, 0) if in_box else (255, 255, 255)
            cv2.rectangle(frame, (self.BOX_X1, self.BOX_Y1), (self.BOX_X2, self.BOX_Y2), box_color, 2)
            
            current_time = time.time()
            
            # Timer & Snapshot Logic
            if current_time < self.cooldown_until:
                cv2.putText(frame, "Snapshot Captured! Resetting...", (self.BOX_X1 + 10, self.BOX_Y1 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self.snapshot_timer_start = None
            else:
                if in_box and not np.all(features == 0.0):
                    if self.snapshot_timer_start is None:
                        self.snapshot_timer_start = current_time
                        
                    remaining = max(0.0, 2.0 - (current_time - self.snapshot_timer_start))
                    
                    if remaining <= 0:
                        # Capture happens here!
                        pred_char, _ = self.get_prediction(features)
                        self.trigger_snapshot(pred_char)
                        self.snapshot_timer_start = None
                    else:
                        cv2.putText(frame, f"Capturing in {remaining:.1f}...", (self.BOX_X1 + 10, self.BOX_Y1 + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    self.snapshot_timer_start = None
                    cv2.putText(frame, "Place hands in box", (self.BOX_X1 + 10, self.BOX_Y1 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                
            # Flash effect
            if self.flash_frames > 0:
                frame[:] = 255
                self.flash_frames -= 1
            
            # Render to Tkinter
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.vid_label.imgtk = imgtk
            self.vid_label.configure(image=imgtk)
            
        self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ISLApp(root)
    root.mainloop()
