import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json
import os
import time
import threading
from collections import deque, Counter
from extractor import MediaPipeExtractor
from translator import SignTranslator
from tts import SpeechEngine

class ISLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Sign2Txt - Detection, Translation & Speech")
        self.root.geometry("860x960")
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

        # Initialize Translation and Speech Engines
        self.translator = SignTranslator()
        self.speech_engine = SpeechEngine()

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
        # Highly responsive buffer: 8 frames (~200-250ms window)
        self.pred_buffer = deque(maxlen=8)
        self.min_consensus_count = 5      # 5 of 8 frames (~62%) to lock in quickly
        self.min_confidence = 0.50         # Responsive confidence threshold
        self.locked_symbol = None
        self.continuous_cooldown_until = 0.0
        self.idle_start_time = None
        self.auto_space_delay = 1.6
        self.auto_space_paused_until = 0.0
        
        self.setup_ui()
        self.update_frame()
        
    def setup_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#1e1e1e")
        header_frame.pack(fill=tk.X, padx=20, pady=(8, 4))
        
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
        self.vid_label.pack(pady=3)
        
        # Mode Explanation Banner
        self.lbl_mode_info = tk.Label(
            self.root, 
            text="[Snapshot Mode] Hold your sign inside the box for 2 seconds to capture.",
            font=("Helvetica", 10, "italic"), fg="#bbbbbb", bg="#1e1e1e"
        )
        self.lbl_mode_info.pack(pady=(0, 2))
        
        # Section 1: Recognized Text
        text_label = tk.Label(
            self.root, text="Recognized English Text:", 
            font=("Helvetica", 10, "bold"), fg="#b0bec5", bg="#1e1e1e"
        )
        text_label.pack(anchor=tk.W, padx=20, pady=(2, 0))

        self.text_box = tk.Text(
            self.root, height=2, width=48, 
            font=("Helvetica", 18, "bold"), bg="#f8f9fa", fg="#212529",
            bd=2, relief=tk.SUNKEN, wrap=tk.WORD
        )
        self.text_box.pack(pady=(0, 4), padx=20)
        
        # Control Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=2)
        
        self.btn_space = tk.Button(
            btn_frame, text="Space", font=("Helvetica", 11, "bold"), 
            bg="#2e7d32", fg="white", activebackground="#1b5e20", activeforeground="white",
            width=8, command=self.add_space
        )
        self.btn_space.grid(row=0, column=0, padx=4, pady=2, ipady=2)
        
        self.btn_backspace = tk.Button(
            btn_frame, text="Backspace", font=("Helvetica", 11, "bold"), 
            bg="#f57c00", fg="white", activebackground="#e65100", activeforeground="white",
            width=10, command=self.backspace
        )
        self.btn_backspace.grid(row=0, column=1, padx=4, pady=2, ipady=2)
        
        self.btn_clear = tk.Button(
            btn_frame, text="Clear All", font=("Helvetica", 11, "bold"), 
            bg="#d32f2f", fg="white", activebackground="#b71c1c", activeforeground="white",
            width=9, command=self.clear_text
        )
        self.btn_clear.grid(row=0, column=2, padx=4, pady=2, ipady=2)
        
        self.btn_save = tk.Button(
            btn_frame, text="Save Text", font=("Helvetica", 11, "bold"), 
            bg="#1976d2", fg="white", activebackground="#0d47a1", activeforeground="white",
            width=10, command=self.save_file
        )
        self.btn_save.grid(row=0, column=3, padx=4, pady=2, ipady=2)
        
        self.btn_quit = tk.Button(
            btn_frame, text="Quit", font=("Helvetica", 11, "bold"), 
            bg="#616161", fg="white", activebackground="#424242", activeforeground="white",
            width=7, command=self.quit_app
        )
        self.btn_quit.grid(row=0, column=4, padx=4, pady=2, ipady=2)

        # Section 2: AI Translation & Speech Panel
        trans_frame = tk.LabelFrame(
            self.root, text=" AI Translation & Speech (Gemini) ", 
            font=("Helvetica", 11, "bold"), fg="#81c784", bg="#1e1e1e", bd=1
        )
        trans_frame.pack(fill=tk.X, padx=20, pady=6)
        
        # Translation Controls Row
        ctrl_row = tk.Frame(trans_frame, bg="#1e1e1e")
        ctrl_row.pack(fill=tk.X, padx=10, pady=4)
        
        tk.Label(
            ctrl_row, text="Target Language:", font=("Helvetica", 10, "bold"), 
            fg="#e0e0e0", bg="#1e1e1e"
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        self.combo_lang = ttk.Combobox(
            ctrl_row, values=[
                "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Bengali", 
                "Marathi", "Gujarati", "Spanish", "French", "German", "Japanese"
            ], state="readonly", width=12, font=("Helvetica", 10)
        )
        self.combo_lang.set("Hindi")
        self.combo_lang.pack(side=tk.LEFT, padx=4)
        
        self.btn_translate = tk.Button(
            ctrl_row, text="✨ AI Translate & Fix", font=("Helvetica", 10, "bold"),
            bg="#0288d1", fg="white", activebackground="#01579b", activeforeground="white",
            command=self.run_translation
        )
        self.btn_translate.pack(side=tk.LEFT, padx=6)
        
        self.btn_speak_trans = tk.Button(
            ctrl_row, text="🔊 Speak (Translation)", font=("Helvetica", 10, "bold"),
            bg="#7b1fa2", fg="white", activebackground="#4a148c", activeforeground="white",
            command=self.speak_translation
        )
        self.btn_speak_trans.pack(side=tk.LEFT, padx=4)

        self.btn_speak_orig = tk.Button(
            ctrl_row, text="🔊 Speak (English)", font=("Helvetica", 10, "bold"),
            bg="#455a64", fg="white", activebackground="#263238", activeforeground="white",
            command=self.speak_original
        )
        self.btn_speak_orig.pack(side=tk.LEFT, padx=4)
        
        # Translation Status Label
        self.lbl_trans_status = tk.Label(
            trans_frame, text="Ready — Click 'AI Translate & Fix' to correct typos and translate.", 
            font=("Helvetica", 9, "italic"), fg="#aaaaaa", bg="#1e1e1e"
        )
        self.lbl_trans_status.pack(anchor=tk.W, padx=10, pady=(1, 2))
        
        # Translated Output Text Box
        self.trans_text_box = tk.Text(
            trans_frame, height=2, width=48, 
            font=("Helvetica", 18, "bold"), bg="#263238", fg="#80cbc4",
            bd=2, relief=tk.SUNKEN, wrap=tk.WORD
        )
        self.trans_text_box.pack(fill=tk.X, padx=10, pady=(2, 6))

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
            
        self.snapshot_timer_start = None
        self.cooldown_until = 0.0
        self.pred_buffer.clear()
        self.locked_symbol = None
        self.continuous_cooldown_until = 0.0
        self.idle_start_time = None
        self.flash_frames = 0

    def add_space(self):
        current = self.text_box.get("1.0", "end-1c")
        if current and not current.endswith(" "):
            self.text_box.insert(tk.END, " ")
            self.text_box.see(tk.END)
        self.idle_start_time = time.time()
        
    def clear_text(self):
        self.text_box.delete("1.0", tk.END)
        self.trans_text_box.delete("1.0", tk.END)
        self.idle_start_time = time.time()
        self.auto_space_paused_until = time.time() + 2.5
        self.lbl_trans_status.config(text="Ready", fg="#aaaaaa")
        
    def backspace(self):
        content = self.text_box.get("1.0", "end-1c")
        if len(content) > 0:
            self.text_box.delete("end-2c", "end-1c")
            self.text_box.see(tk.END)
        self.auto_space_paused_until = time.time() + 2.5
        self.idle_start_time = time.time()
            
    def save_file(self):
        fp = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if fp:
            try:
                with open(fp, "w", encoding="utf-8") as f:
                    f.write("=== Recognized English ===\n")
                    f.write(self.text_box.get("1.0", tk.END).strip() + "\n\n")
                    f.write(f"=== Translation ({self.combo_lang.get()}) ===\n")
                    f.write(self.trans_text_box.get("1.0", tk.END).strip() + "\n")
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
        if pred_char.lower() == "space":
            self.add_space()
        elif pred_char.lower() in ["delete", "del", "backspace"]:
            self.backspace()
        else:
            self.text_box.insert(tk.END, pred_char)
            self.text_box.see(tk.END)
            
        self.flash_frames = 2 if is_continuous else 4
        if not is_continuous:
            self.cooldown_until = time.time() + 1.5

    def check_auto_space(self, current_time):
        if hasattr(self, "auto_space_paused_until") and current_time < self.auto_space_paused_until:
            return "Auto-Space Paused"
            
        if self.idle_start_time is None:
            self.idle_start_time = current_time
            return None
            
        idle_duration = current_time - self.idle_start_time
        remaining = max(0.0, self.auto_space_delay - idle_duration)
        
        if remaining <= 0.0:
            current_text = self.text_box.get("1.0", "end-1c")
            if current_text and not current_text.endswith(" "):
                self.text_box.insert(tk.END, " ")
                self.text_box.see(tk.END)
                self.flash_frames = 1
            self.idle_start_time = current_time
            return "Auto-Spaced"
        return f"Auto-Space in {remaining:.1f}s"

    def handle_continuous_mode(self, frame, in_box, features, current_time, pred_char, conf):
        if not in_box or np.all(features == 0.0):
            self.pred_buffer.clear()
            self.locked_symbol = None
            status_text = self.check_auto_space(current_time)
            msg = status_text if status_text else "Place hands in box"
            cv2.putText(frame, msg, (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
            return

        self.idle_start_time = None
        
        if current_time < self.continuous_cooldown_until:
            cv2.putText(frame, "Refractory Cooldown...", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            return
            
        # Re-use pre-calculated prediction (zero redundant inference!)
        if conf >= self.min_confidence and pred_char != "None":
            self.pred_buffer.append(pred_char)
        else:
            self.pred_buffer.append(None)

        if self.locked_symbol is not None:
            locked_occurrences = self.pred_buffer.count(self.locked_symbol)
            if locked_occurrences < 2:
                self.locked_symbol = None

        valid_preds = [p for p in self.pred_buffer if p is not None]
        
        if len(valid_preds) >= 4:
            candidate, count = Counter(valid_preds).most_common(1)[0]
            
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
                cv2.putText(frame, f"Locked: {candidate} (Change sign to continue)", 
                            (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            elif count >= self.min_consensus_count:
                self.trigger_char_emit(candidate, is_continuous=True)
                self.locked_symbol = candidate
                self.pred_buffer.clear()
                self.continuous_cooldown_until = current_time + 0.25
            else:
                cv2.putText(frame, f"Stabilizing: {candidate} ({count}/{self.min_consensus_count})", 
                            (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
        else:
            cv2.putText(frame, "Analyzing Gesture...", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 2)

    def handle_snapshot_mode(self, frame, in_box, features, current_time, pred_char, conf):
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
                
                prog = min(1.0, elapsed / 2.0)
                bar_w = int((self.BOX_X2 - self.BOX_X1 - 20) * prog)
                cv2.rectangle(
                    frame, 
                    (self.BOX_X1 + 10, self.BOX_Y2 - 22), 
                    (self.BOX_X1 + 10 + bar_w, self.BOX_Y2 - 10), 
                    (0, 255, 0), -1
                )
                
                if remaining <= 0:
                    self.trigger_char_emit(pred_char, is_continuous=False)
                    self.snapshot_timer_start = None
                else:
                    cv2.putText(frame, f"Capturing in {remaining:.1f}...", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                self.snapshot_timer_start = None
                cv2.putText(frame, "Place hands in box", (self.BOX_X1 + 10, self.BOX_Y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- Translation and Speech Handlers ---
    def run_translation(self):
        raw_text = self.text_box.get("1.0", "end-1c").strip()
        if not raw_text:
            self.lbl_trans_status.config(text="Nothing to translate! Sign some letters first.", fg="#ff9800")
            return
            
        target_lang = self.combo_lang.get()
        self.lbl_trans_status.config(text=f"Translating to {target_lang} with Gemini AI...", fg="#29b6f6")
        self.btn_translate.config(state=tk.DISABLED)
        
        def _worker():
            res = self.translator.translate(raw_text, target_language=target_lang)
            self.root.after(0, lambda: self._on_translation_done(res))
            
        threading.Thread(target=_worker, daemon=True).start()
        
    def _on_translation_done(self, res):
        self.btn_translate.config(state=tk.NORMAL)
        corrected = res.get("corrected_english", "")
        translated = res.get("translated_text", "")
        engine = res.get("engine", "")
        
        # Update English text box with AI-repaired version if available
        if corrected and engine == "gemini":
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, corrected)
            self.text_box.see(tk.END)
            
        self.trans_text_box.delete("1.0", tk.END)
        self.trans_text_box.insert(tk.END, translated)
        self.trans_text_box.see(tk.END)
        
        if engine == "gemini":
            self.lbl_trans_status.config(text="Translated & Typos Fixed via Gemini 3.6 Flash! ✨", fg="#66bb6a")
        elif engine == "fallback":
            self.lbl_trans_status.config(text="Translated via Google Translator fallback.", fg="#ffd54f")
        else:
            self.lbl_trans_status.config(text="Translation error. Check connection.", fg="#ef5350")
            
    def speak_translation(self):
        text = self.trans_text_box.get("1.0", "end-1c").strip()
        if text:
            self.speech_engine.speak(text)
        else:
            self.lbl_trans_status.config(text="No translated text to speak!", fg="#ff9800")
            
    def speak_original(self):
        text = self.text_box.get("1.0", "end-1c").strip()
        if text:
            self.speech_engine.speak(text)
        else:
            self.lbl_trans_status.config(text="No English text to speak!", fg="#ff9800")

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.consecutive_cam_fails += 1
                if self.consecutive_cam_fails > 10:
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
            
            try:
                features, results = self.extractor.extract(frame)
                frame = self.extractor.draw_landmarks(frame, results)
            except Exception as e:
                print(f"Extractor exception: {e}")
                features = np.zeros(132, dtype=np.float32)
                results = None
            
            in_box = False
            if results and results.hand_landmarks:
                in_box = True
                for hand_landmarks in results.hand_landmarks:
                    wrist = hand_landmarks[0]
                    wx, wy = int(wrist.x * self.WIDTH), int(wrist.y * self.HEIGHT)
                    if not (self.BOX_X1 <= wx <= self.BOX_X2 and self.BOX_Y1 <= wy <= self.BOX_Y2):
                        in_box = False
                        break
                        
            pred_char, conf = "None", 0.0
            display_text = "Prediction: None"
            color = (120, 120, 120)
            if not np.all(features == 0.0):
                pred_char, conf = self.get_prediction(features)
                display_text = f"Live: {pred_char} ({conf*100:.1f}%)"
                color = (0, 255, 0) if conf > 0.50 else (0, 165, 255)
                
            cv2.putText(frame, display_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            box_color = (0, 255, 0) if in_box else (100, 100, 100)
            cv2.rectangle(frame, (self.BOX_X1, self.BOX_Y1), (self.BOX_X2, self.BOX_Y2), box_color, 2)
            
            current_time = time.time()
            
            if self.mode == "snapshot":
                self.handle_snapshot_mode(frame, in_box, features, current_time, pred_char, conf)
            else:
                self.handle_continuous_mode(frame, in_box, features, current_time, pred_char, conf)
                                
            if self.flash_frames > 0:
                frame[:] = 255
                self.flash_frames -= 1
            
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.vid_label.imgtk = imgtk
            self.vid_label.configure(image=imgtk)

        except Exception as e:
            print(f"Unexpected loop exception: {e}")
            
        self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ISLApp(root)
    root.mainloop()
