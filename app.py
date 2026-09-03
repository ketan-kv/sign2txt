import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json
import os
import time
from image_utils import preprocess_for_cnn

class ISLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Detection - Tier-1 Edge CNN")
        self.root.geometry("800x800")
        self.root.configure(bg="#2b2b2b")
        
        if not os.path.exists("isl_cnn_model.keras") or not os.path.exists("labels.json"):
            print("Error: Required model files missing. Run train_cnn.py first.")
            self.root.destroy()
            return
            
        self.model = tf.keras.models.load_model("isl_cnn_model.keras")
        with open("labels.json", "r") as f:
            raw_map = json.load(f)
            self.class_names = {int(k): v for k, v in raw_map.items()}
            
        self.cap = cv2.VideoCapture(0)
        
        self.snapshot_timer_start = None
        self.cooldown_until = 0.0
        self.flash_frames = 0
        
        self.WIDTH, self.HEIGHT = 640, 480
        self.CNN_IMG_SIZE = 128 # Matching the new custom CNN
        
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
        self.root.destroy()
        
    def trigger_snapshot(self, pred_char):
        if pred_char.lower() == "space":
            self.text_box.insert(tk.END, " ")
        elif pred_char.lower() in ["delete", "del", "backspace"]:
            self.backspace()
        else:
            self.text_box.insert(tk.END, pred_char)
            
        self.flash_frames = 3
        self.cooldown_until = time.time() + 1.5
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            
            # Extract crop and apply 1-Channel Canny Edge preprocessing
            hand_crop = frame[self.BOX_Y1:self.BOX_Y2, self.BOX_X1:self.BOX_X2]
            input_data = preprocess_for_cnn(hand_crop, img_size=self.CNN_IMG_SIZE)
            
            # Predict
            # input_data is (128, 128, 1), Keras expects batch dim (1, 128, 128, 1)
            predictions = self.model(np.expand_dims(input_data, axis=0), training=False).numpy()[0]
            max_idx = np.argmax(predictions)
            conf = predictions[max_idx]
            pred_char = self.class_names[max_idx]
            
            # Live Debug Info
            display_text = f"CNN: {pred_char} ({conf*100:.1f}%)"
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (self.BOX_X1, self.BOX_Y1), (self.BOX_X2, self.BOX_Y2), color, 2)
            
            current_time = time.time()
            if current_time < self.cooldown_until:
                cv2.putText(frame, "Snapshot Captured! Resetting...", (self.BOX_X1 + 10, self.BOX_Y1 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self.snapshot_timer_start = None
            else:
                if conf > 0.7:
                    if self.snapshot_timer_start is None:
                        self.snapshot_timer_start = current_time
                        
                    remaining = max(0.0, 2.0 - (current_time - self.snapshot_timer_start))
                    
                    if remaining <= 0:
                        self.trigger_snapshot(pred_char)
                        self.snapshot_timer_start = None
                    else:
                        cv2.putText(frame, f"Capturing in {remaining:.1f}...", (self.BOX_X1 + 10, self.BOX_Y1 + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    self.snapshot_timer_start = None
                    cv2.putText(frame, "Hold sign in box clearly", (self.BOX_X1 + 10, self.BOX_Y1 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                
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
