import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json
import os
import time
from collections import deque, Counter
from extractor import MediaPipeExtractor

class ISLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Sign2Txt - Multi-Mode Detection")
        self.root.geometry("820x860")
        self.root.configure(bg="#1e1e1e")
        
        # 1. Defensive Startup Checks
        if not os.path.exists("isl_dnn_model.keras") or not os.path.exists("labels.json"):
            messagebox.showerror(
                "Model Not Found", 
                "Required model files (isl_dnn_model.keras / labels.json) are missing.\nPlease run train.py first."
            )
            self.root.destroy()
            return
            
        try:
            self.model = tf.keras.models.load_model("isl_dnn_model.keras")
            # Verify input shape
            expected_features = 132
            if self.model.input_shape[-1] != expected_features:
                messagebox.showerror(
                    "Model Shape Error",
                    f"Expected model input shape of {expected_features} features, but got {self.model.input_shape[-1]}."
                )
                self.root.destroy()
                return
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load isl_dnn_model.keras:\n{e}")
            self.root.destroy()
            return

        try:
            with open("labels.json", "r") as f:
                self.class_names = json.load(f)
        except Exception as e:
            messagebox.showerror("Labels Load Error", f"Failed to load labels.json:\n{e}")
            self.root.destroy()
            return
            
        try:
            self.extractor = MediaPipeExtractor()
        except Exception as e:
            messagebox.showerror("MediaPipe Error", f"Failed to initialize MediaPipeExtractor:\n{e}")
            self.root.destroy()
            return

        self.cap = cv2.VideoCapture(0)
        self.consecutive_cam_fails = 0
        
        # UI & Camera Constants
        self.WIDTH, self.HEIGHT = 640, 480
        self.BOX_X1 = int(self.WIDTH * 0.15)
        self.BOX_Y1 = int(self.HEIGHT * 0.15)
        self.BOX_X2 = int(self.WIDTH * 0.85)
        self.BOX_Y2 = int(self.HEIGHT * 0.85)
        
        # Mode Management: "snapshot" or "continuous"
        self.mode = "snapshot"
        
        # --- Snapshot Mode State ---
        self.snapshot_timer_start = None
        self.cooldown_until = 0.0
        self.flash_frames = 0
        
        # --- Continuous Mode State Machine ---
        # Buffer size of 12 frames (~350-400ms window)
        self.pred_buffer = deque(maxlen=12)
        self.min_consensus_count = 9      # At least 9 of 12 frames (~75%) must agree
        self.min_confidence = 0.65         # Minimum confidence to accept a frame
        self.locked_symbol = None          # Prevents runaway spam (e.g. AAAAA)
        self.continuous_cooldown_until = 0.0
        self.idle_start_time = None        # Tracks absence of hands for auto-space
        self.auto_space_delay = 1.6        # 1.6 seconds without hands triggers space
        
        self.setup_ui()
        self.update_frame()
        
    def setup_ui(self):
        # Header / Status Frame
        header_frame = tk.Frame(self.root, bg="#1e1e1e")
        header_frame.pack(fill=tk.X, padx=20, pady=(10, 5))
        
        title_label = tk.Label(
            header_frame, text="ISL Sign2Txt Assistant", 
            font=("Helvetica", 16, "bold"), fg="#ffffff", bg="#1e1e1e"
        )
        title_label.pack(side=tk.LEFT)
        
        # Mode Toggle Button
        self.btn_mode = tk.Button(
            header_frame, text="Mode: Snapshot (Click to Switch)", 
            font=("Helvetica", 11, "bold"), bg="#673ab7", fg="white", 
            activebackground="#512da8", activeforeground="white",
            relief=tk.RAISED, cursor="hand2", command=self.toggle_mode
        )
        self.btn_mode.pack(side=tk.RIGHT, ipadx=10, ipady=3)
        
        # Video Display Label
        self.vid_label = tk.Label(self.root, bg="#111111", bd=2, relief=tk.SOLID)
        self.vid_label.pack(pady=5)
        
        # Mode Explanation Banner
        self.lbl_mode_info = tk.Label(
            self.root, 
            text="[Snapshot Mode] Hold your sign inside the box for 2 seconds to capture.",
            font=("Helvetica", 11, "italic"), fg="#bbbbbb", bg="#1e1e1e"
        )
        self.lbl_mode_info.pack(pady=(0, 5))
        
        # Predicted Text Output Box
        self.text_box = tk.Text(
            self.root, height=3, width=42, 
            font=("Helvetica", 22, "bold"), bg="#f8f9fa", fg="#212529",
            bd=2, relief=tk.SUNKEN, wrap=tk.WORD
        )
        self.text_box.pack(pady=5, padx=20)
        
        # Control Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=10)
        
        self.btn_space = tk.Button(
            btn_frame, text="Space", font=("Helvetica", 12, "bold"), 
            bg="#2e7d32", fg="white", activebackground="#1b5e20", activeforeground="white",
            width=10, command=self.add_space
        )
        self.btn_space.grid(row=0, column=0, padx=6, pady=5, ipady=4)
        
        self.btn_backspace = tk.Button(
            btn_frame, text="Backspace", font=("Helvetica", 12, "bold"), 
            bg="#f57c00", fg="white", activebackground="#e65100", activeforeground="white",
            width=10, command=self.backspace
        )
        self.btn_backspace.grid(row=0, column=1, padx=6, pady=5, ipady=4)
        
        self.btn_clear = tk.Button(
            btn_frame, text="Clear All", font=("Helvetica", 12, "bold"), 
            bg="#d32f2f", fg="white", activebackground="#b71c1c", activeforeground="white",
            width=10, command=self.clear_text
        )
        self.btn_clear.grid(row=0, column=2, padx=6, pady=5, ipady=4)
        
        self.btn_save = tk.Button(
            btn_frame, text="Save to File", font=("Helvetica", 12, "bold"), 
            bg="#1976d2", fg="white", activebackground="#0d47a1", activeforeground="white",
            width=12, command=self.save_file
        )
        self.btn_save.grid(row=0, column=3, padx=6, pady=5, ipady=4)
        
        self.btn_quit = tk.Button(
            btn_frame, text="Quit", font=("Helvetica", 12, "bold"), 
            bg="#616161", fg="white", activebackground="#424242", activeforeground="white",
            width=8, command=self.quit_app
        )
        self.btn_quit.grid(row=0, column=4, padx=6, pady=5, ipady=4)

    def toggle_mode(self):
        """Flushes states and toggles between Snapshot and Continuous mode."""
        if self.mode == "snapshot":
            self.mode = "continuous"
            self.btn_mode.config(
                text="Mode: Continuous (Click to Switch)", 
                bg="#00897b", activebackground="#00695c"
            )
            self.lbl_mode_info.config(
                text="[Continuous Mode] Sign continuously. Hands-free space triggers after 1.5s idle."
            )
        else:
            self.mode = "snapshot"
            self.btn_mode.config(
                text="Mode: Snapshot (Click to Switch)", 
                bg="#673ab7", activebackground="#512da8"
            )
            self.lbl_mode_info.config(
                text="[Snapshot Mode] Hold your sign inside the box for 2 seconds to capture."
            )
            
        # Flush all state variables to prevent cross-mode pollution
        self.snapshot_timer_start = None
        self.cooldown_until = 0.0
        self.pred_buffer.clear()
        self.locked_symbol = None
        self.continuous_cooldown_until = 0.0
        self.idle_start_time = None
        self.flash_frames = 0

    def add_space(self):
        self.text_box.insert(tk.END, " ")
        self.text_box.see(tk.END)
        
    def clear_text(self):
        self.text_box.delete("1.0", tk.END)
        
    def backspace(self):
        current = self.text_box.get("1.0", tk.END)
        if len(current) > 1:
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, current[:-2])
            self.text_box.see(tk.END)
            
    def save_file(self):
        fp = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if fp:
            try:
                with open(fp, "w") as f:
                    f.write(self.text_box.get("1.0", tk.END))
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save file:\n{e}")
                
    def quit_app(self):
        try:
            self.cap.release()
        except:
            pass
        try:
            self.extractor.detector.close()
        except:
            pass
        self.root.destroy()
        
    def get_prediction(self, features):
        try:
            input_data = np.expand_dims(features, axis=0)
            predictions = self.model(input_data, training=False).numpy()
            max_idx = np.argmax(predictions[0])
            return self.class_names.get(str(max_idx), str(max_idx)), float(predictions[0][max_idx])
        except Exception as e:
            print(f"Prediction error: {e}")
            return "None", 0.0
        
    def trigger_char_emit(self, pred_char, is_continuous=False):
        """Appends character to text box and handles special control characters."""
        if pred_char.lower() == "space":
            self.add_space()
        elif pred_char.lower() in ["delete", "del", "backspace"]:
            self.backspace()
        else:
            self.text_box.insert(tk.END, pred_char)
            self.text_box.see(tk.END)
            
        # Visual flash indicator
        self.flash_frames = 2 if is_continuous else 4
        if not is_continuous:
            self.cooldown_until = time.time() + 1.5

    def check_auto_space(self, current_time):
        """Automatically appends space when hands remain absent for auto_space_delay."""
        if self.idle_start_time is None:
            self.idle_start_time = current_time
            return None
            
        idle_duration = current_time - self.idle_start_time
        remaining = max(0.0, self.auto_space_delay - idle_duration)
        
        if remaining <= 0.0:
            current_text = self.text_box.get("1.0", tk.END).rstrip("\n")
            if current_text and not current_text.endswith(" "):
                self.add_space()
                self.flash_frames = 1
            # Reset timer so we don't spam spaces indefinitely
            self.idle_start_time = current_time
            return "Auto-Spaced"
        return f"Auto-Space in {remaining:.1f}s"

    def handle_continuous_mode(self, frame, in_box, features, current_time):
        """Manages the 5-state temporal consensus engine for stream signing."""
        if not in_box or np.all(features == 0.0):
            # State: IDLE / Hands Absent
            self.pred_buffer.clear()
            self.locked_symbol = None
            status_text = self.check_auto_space(current_time)
            msg = status_text if status_text else "Place hands in box"
            cv2.putText(frame, msg, (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
            return

        # Hands present: reset idle auto-space timer
        self.idle_start_time = None
        
        # State: COOLDOWN (Refractory period right after an emission)
        if current_time < self.continuous_cooldown_until:
            cv2.putText(frame, "Refractory Cooldown...", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            return
            
        # State: STABILIZING (Collecting frame predictions)
        pred_char, conf = self.get_prediction(features)
        
        if conf >= self.min_confidence and pred_char != "None":
            self.pred_buffer.append(pred_char)
        else:
            self.pred_buffer.append(None) # Represents an ambiguous/transitional frame

        # Check unlocked condition: if buffer has moved away from the locked symbol
        if self.locked_symbol is not None:
            locked_occurrences = self.pred_buffer.count(self.locked_symbol)
            # If the locked symbol is now rare in the buffer (< 3 frames), unlock it!
            if locked_occurrences < 3:
                self.locked_symbol = None

        # Compute consensus from valid predictions in the buffer
        valid_preds = [p for p in self.pred_buffer if p is not None]
        
        if len(valid_preds) >= 6:
            candidate, count = Counter(valid_preds).most_common(1)[0]
            
            # Draw live consensus progress bar
            progress_ratio = min(1.0, count / float(self.min_consensus_count))
            bar_w = int((self.BOX_X2 - self.BOX_X1 - 20) * progress_ratio)
            cv2.rectangle(
                frame, 
                (self.BOX_X1 + 10, self.BOX_Y2 - 22), 
                (self.BOX_X1 + 10 + bar_w, self.BOX_Y2 - 10), 
                (0, 220, 0) if count >= self.min_consensus_count else (0, 180, 255), 
                -1
            )
            
            if candidate == self.locked_symbol:
                # State: LOCKED (Prevent duplicate spam)
                cv2.putText(frame, f"Locked: {candidate} (Change sign to continue)", 
                            (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            elif count >= self.min_consensus_count:
                # State: EMIT
                self.trigger_char_emit(candidate, is_continuous=True)
                self.locked_symbol = candidate
                self.pred_buffer.clear()
                self.continuous_cooldown_until = current_time + 0.35
            else:
                cv2.putText(frame, f"Stabilizing: {candidate} ({count}/{self.min_consensus_count})", 
                            (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
        else:
            cv2.putText(frame, "Analyzing Gesture...", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 2)

    def handle_snapshot_mode(self, frame, in_box, features, current_time):
        """Manages the proven 2.0s countdown Snapshot Mode from Stage 1."""
        if current_time < self.cooldown_until:
            cv2.putText(frame, "Snapshot Captured! Resetting...", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self.snapshot_timer_start = None
        else:
            if in_box and not np.all(features == 0.0):
                if self.snapshot_timer_start is None:
                    self.snapshot_timer_start = current_time
                    
                elapsed = current_time - self.snapshot_timer_start
                remaining = max(0.0, 2.0 - elapsed)
                
                # Draw countdown progress bar
                prog = min(1.0, elapsed / 2.0)
                bar_w = int((self.BOX_X2 - self.BOX_X1 - 20) * prog)
                cv2.rectangle(
                    frame, 
                    (self.BOX_X1 + 10, self.BOX_Y2 - 22), 
                    (self.BOX_X1 + 10 + bar_w, self.BOX_Y2 - 10), 
                    (0, 255, 0), -1
                )
                
                if remaining <= 0:
                    pred_char, _ = self.get_prediction(features)
                    self.trigger_char_emit(pred_char, is_continuous=False)
                    self.snapshot_timer_start = None
                else:
                    cv2.putText(frame, f"Capturing in {remaining:.1f}...", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                self.snapshot_timer_start = None
                cv2.putText(frame, "Place hands in box", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.consecutive_cam_fails += 1
                if self.consecutive_cam_fails > 10:
                    # Draw camera disconnected graphic
                    blank = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
                    cv2.putText(blank, "CAMERA NOT AVAILABLE / DISCONNECTED", (30, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    img = Image.fromarray(blank)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.vid_label.imgtk = imgtk
                    self.vid_label.configure(image=imgtk)
                self.root.after(30, self.update_frame)
                return

            self.consecutive_cam_fails = 0
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            
            # Landmark extraction & rendering
            try:
                features, results = self.extractor.extract(frame)
                frame = self.extractor.draw_landmarks(frame, results)
            except Exception as e:
                print(f"Extractor exception: {e}")
                features = np.zeros(132, dtype=np.float32)
                results = None
            
            # Check if all detected hands are within target bounding box
            in_box = False
            if results and results.hand_landmarks:
                in_box = True
                for hand_landmarks in results.hand_landmarks:
                    wrist = hand_landmarks[0]
                    wx, wy = int(wrist.x * self.WIDTH), int(wrist.y * self.HEIGHT)
                    if not (self.BOX_X1 <= wx <= self.BOX_X2 and self.BOX_Y1 <= wy <= self.BOX_Y2):
                        in_box = False
                        break
                        
            # Real-time top display overlay
            display_text = "Prediction: None"
            color = (120, 120, 120)
            if not np.all(features == 0.0):
                pred_char, conf = self.get_prediction(features)
                display_text = f"Live: {pred_char} ({conf*100:.1f}%)"
                color = (0, 255, 0) if conf > 0.65 else (0, 165, 255)
                
            cv2.putText(frame, display_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw Target Box outline
            box_color = (0, 255, 0) if in_box else (100, 100, 100)
            cv2.rectangle(frame, (self.BOX_X1, self.BOX_Y1), (self.BOX_X2, self.BOX_Y2), box_color, 2)
            
            current_time = time.time()
            
            # Dispatch to active mode engine
            if self.mode == "snapshot":
                self.handle_snapshot_mode(frame, in_box, features, current_time)
            else:
                self.handle_continuous_mode(frame, in_box, features, current_time)
                                
            # Screen Flash Effect
            if self.flash_frames > 0:
                frame[:] = 255
                self.flash_frames -= 1
            
            # Render frame to Tkinter
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.vid_label.imgtk = imgtk
            self.vid_label.configure(image=imgtk)

        except Exception as e:
            print(f"Unexpected loop exception: {e}")
            
        self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ISLApp(root)
    root.mainloop()
