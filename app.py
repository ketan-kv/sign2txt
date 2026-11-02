"""Live Indian Sign Language -> text -> translation window.

Usage:   python app.py [--camera 0] [--width 720]
Set your Gemini key with the GEMINI_API_KEY environment variable, or paste it in Settings.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import tkinter as tk

import customtkinter as ctk
import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

from common import (CLASSIFIER_PATH, HAND_CONNECTIONS, create_landmarker, detect_hands,
                    hands_to_features)
from composer import SignComposer
from speech import Player, SpeechError, synthesize

LANGUAGES = ["Hindi", "English", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam",
             "Marathi", "Gujarati", "Punjabi", "Urdu", "Odia", "Assamese", "Spanish",
             "French", "German", "Japanese", "Chinese", "Arabic"]

PROMPT = """You are helping translate Indian Sign Language (ISL) finger-spelling.
The text below came from a letter-by-letter recognizer, so it may contain wrong, missing or
extra letters and imperfect word breaks. It may be English, or Indian words / names written
in Latin letters (e.g. Hinglish).

Step 1: infer the most likely intended sentence (fix obvious spelling errors; do not add new ideas).
Step 2: translate that sentence into {language}.

Return ONLY JSON with keys:
  "corrected":   the cleaned-up sentence in its original language,
  "translation": the translation into {language},
  "romanized":   a Latin-script pronunciation of the translation if {language} does not use
                 the Latin alphabet, otherwise "".

Recognized text: {text}"""

# palette
BG, CARD, CARD2 = "#0e1116", "#171b22", "#222836"
ACCENT, OK, WARN, MUTED, TEXT = "#4f8cff", "#2ecc71", "#f5a623", "#8b93a1", "#e8ecf2"


def F(size, weight="normal"):
    return ctk.CTkFont(size=size, weight=weight)


class App:
    def __init__(self, root: ctk.CTk, args):
        self.root = root
        self.disp_w = args.width
        self.disp_h = int(args.width * 3 / 4)
        root.title("ISL Translator")
        root.configure(fg_color=BG)

        bundle = joblib.load(CLASSIFIER_PATH)
        self.pipe, self.classes = bundle["pipeline"], bundle["classes"]
        self.landmarker = create_landmarker(video=True, num_hands=2, min_det=0.5, min_track=0.5)
        self.composer = SignComposer(conf_thresh=args.conf, hold_s=args.hold,
                                     word_gap_s=args.word_gap)
        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

        self.last_ts = 0
        self.fps = 0.0
        self._t_prev = time.perf_counter()
        self._pill_at = 0.0
        self.flash_until = 0.0
        self.translating = False
        self.last_translation = ""
        self.results: queue.Queue = queue.Queue()      # worker threads -> UI thread
        self.last_language = "English"
        self.player = Player()
        self.speaking = False                          # a Speak request is in progress
        self._playing = False                          # audio is actually playing
        self.speech_token = 0                          # lets us ignore a cancelled request
        self._voice_cache: dict = {}
        self._build_ui()

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", lambda e: self._shortcut(self.word_break))
        root.bind("<BackSpace>", lambda e: self._shortcut(self.backspace))
        root.bind("<Return>", lambda e: self._shortcut(self.translate))
        self._after = root.after(10, self.loop)

    # ------------------------------------------------------------------ UI
    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, **kw)

    def _build_ui(self):
        r = self.root
        r.grid_columnconfigure(1, weight=1)
        r.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(r, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(12, 2))
        ctk.CTkLabel(header, text="ISL Translator", font=F(22, "bold"),
                     text_color=TEXT).pack(side="left")
        ctk.CTkLabel(header, text="  fingerspelling → words → any language", font=F(13),
                     text_color=MUTED).pack(side="left", pady=(5, 0))
        self.pill = ctk.CTkLabel(header, text="● starting…", font=F(13), text_color=MUTED)
        self.pill.pack(side="right")

        # ---- left: camera
        vcard = self._card(r)
        vcard.grid(row=1, column=0, sticky="n", padx=(18, 8), pady=8)
        self.video = tk.Label(vcard, bg=CARD, bd=0, width=self.disp_w, height=self.disp_h)
        self.video.pack(padx=10, pady=(10, 4))
        ctk.CTkLabel(vcard, justify="left", anchor="w", text_color=MUTED, font=F(12),
                     wraplength=self.disp_w,
                     text="Hold a sign until the bar fills to add a letter  ·  lower both hands "
                          "to finish a word  ·  Space / Backspace / Enter = word break / delete / "
                          "translate").pack(fill="x", padx=14, pady=(0, 10))

        # ---- right column
        right = ctk.CTkFrame(r, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 18), pady=8)

        d = self._card(right); d.pack(fill="x", pady=(0, 8))
        d.grid_columnconfigure(1, weight=1)
        self.sign_lbl = ctk.CTkLabel(d, text="–", font=F(58, "bold"), width=96, text_color=MUTED)
        self.sign_lbl.grid(row=0, column=0, rowspan=3, padx=(12, 4), pady=8)
        ctk.CTkLabel(d, text="DETECTED SIGN", font=F(11, "bold"), text_color=MUTED,
                     anchor="w").grid(row=0, column=1, sticky="w", pady=(12, 0))
        self.conf_lbl = ctk.CTkLabel(d, text="Show your hand", font=F(14), anchor="w",
                                     text_color=TEXT, width=330)
        self.conf_lbl.grid(row=1, column=1, sticky="w")
        self.hold_bar = ctk.CTkProgressBar(d, height=10, progress_color=ACCENT, fg_color=CARD2)
        self.hold_bar.set(0)
        self.hold_bar.grid(row=2, column=1, sticky="ew", padx=(0, 16), pady=(2, 14))

        w = self._card(right); w.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(w, text="CURRENT WORD", font=F(11, "bold"), text_color=MUTED,
                     anchor="w").pack(fill="x", padx=16, pady=(10, 0))
        self.word_lbl = ctk.CTkLabel(w, text="", font=F(32, "bold"), anchor="w", text_color=TEXT)
        self.word_lbl.pack(fill="x", padx=16, pady=(0, 8))

        tabs = ctk.CTkTabview(right, fg_color=CARD, corner_radius=16,
                              segmented_button_selected_color=ACCENT)
        tabs.pack(fill="both", expand=True)
        t1, t2 = tabs.add("Translate"), tabs.add("Settings")

        # ---- Translate tab
        ctk.CTkLabel(t1, text="RECOGNIZED TEXT  (you can edit it)", font=F(11, "bold"),
                     text_color=MUTED, anchor="w").pack(fill="x")
        self.sentence = ctk.CTkTextbox(t1, height=78, font=F(17), wrap="word", fg_color=CARD2)
        self.sentence.pack(fill="x", pady=(2, 6))

        row = ctk.CTkFrame(t1, fg_color="transparent"); row.pack(fill="x")
        for text, cmd in [("Word break", self.word_break), ("Backspace", self.backspace),
                          ("Clear all", self.clear)]:
            ctk.CTkButton(row, text=text, width=96, height=30, fg_color=CARD2, hover_color="#2d3547",
                          command=cmd).pack(side="left", padx=(0, 6))

        row2 = ctk.CTkFrame(t1, fg_color="transparent"); row2.pack(fill="x", pady=(12, 6))
        ctk.CTkLabel(row2, text="Translate to", text_color=MUTED).pack(side="left")
        self.lang = ctk.CTkComboBox(row2, values=LANGUAGES, width=150, fg_color=CARD2,
                                    border_width=0, button_color=ACCENT)
        self.lang.set("Hindi"); self.lang.pack(side="left", padx=8)
        self.tr_btn = ctk.CTkButton(row2, text="Translate", height=34, font=F(14, "bold"),
                                    fg_color=ACCENT, command=self.translate)
        self.tr_btn.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(t1, text="TRANSLATION", font=F(11, "bold"), text_color=MUTED,
                     anchor="w").pack(fill="x")
        self.output = ctk.CTkTextbox(t1, height=104, font=F(17), wrap="word", fg_color=CARD2,
                                     state="disabled")
        self.output.pack(fill="x", pady=(2, 4))
        row3 = ctk.CTkFrame(t1, fg_color="transparent"); row3.pack(fill="x")
        ctk.CTkButton(row3, text="Copy", width=70, height=28, fg_color=CARD2,
                      hover_color="#2d3547", command=self.copy_translation).pack(side="left")
        self.speak_btn = ctk.CTkButton(row3, text="Speak", width=80, height=28, fg_color=CARD2,
                                       hover_color="#2d3547", command=self.speak)
        self.speak_btn.pack(side="left", padx=(6, 0))
        self.time_lbl = ctk.CTkLabel(row3, text="", text_color=MUTED, font=F(12))
        self.time_lbl.pack(side="left", padx=10)

        # ---- Settings tab
        ctk.CTkLabel(t2, text="Gemini API key", text_color=MUTED, anchor="w").pack(fill="x")
        self.key = ctk.CTkEntry(t2, show="•", fg_color=CARD2, border_width=0)
        self.key.insert(0, os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""))
        self.key.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(t2, text="Gemini model", text_color=MUTED, anchor="w").pack(fill="x")
        self.model = ctk.CTkEntry(t2, fg_color=CARD2, border_width=0)
        self.model.insert(0, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
        self.model.pack(fill="x", pady=(0, 8))
        self.auto = ctk.CTkSwitch(t2, text="Auto-translate after a long pause", progress_color=ACCENT)
        self.auto.pack(anchor="w", pady=(0, 8))
        self.auto_speak = ctk.CTkSwitch(t2, text="Speak the translation automatically",
                                        progress_color=ACCENT)
        self.auto_speak.pack(anchor="w", pady=(0, 8))
        vrow = ctk.CTkFrame(t2, fg_color="transparent"); vrow.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(vrow, text="Voice", text_color=MUTED).pack(side="left")
        self.voice_menu = ctk.CTkOptionMenu(vrow, values=["Female", "Male"], width=110,
                                            fg_color=CARD2, button_color=ACCENT)
        self.voice_menu.pack(side="left", padx=8)

        c = self.composer
        self._slider(t2, "Hold time to add a letter", 0.4, 1.5, c.hold_s, lambda v: f"{v:.2f} s",
                     lambda v: setattr(c, "hold_s", v))
        self._slider(t2, "Minimum confidence", 0.4, 0.95, c.conf_thresh, lambda v: f"{v * 100:.0f}%",
                     lambda v: setattr(c, "conf_thresh", v))
        self._slider(t2, "Pause that ends a word", 1.0, 3.5, c.word_gap_s, lambda v: f"{v:.1f} s",
                     lambda v: setattr(c, "word_gap_s", v))

    def _slider(self, parent, label, lo, hi, init, fmt, setter):
        box = ctk.CTkFrame(parent, fg_color="transparent"); box.pack(fill="x", pady=3)
        lbl = ctk.CTkLabel(box, text=f"{label}: {fmt(init)}", anchor="w", text_color=TEXT)
        lbl.pack(fill="x")

        def cb(v):
            setter(float(v)); lbl.configure(text=f"{label}: {fmt(float(v))}")
        s = ctk.CTkSlider(box, from_=lo, to=hi, command=cb, progress_color=ACCENT)
        s.set(init); s.pack(fill="x")

    def _shortcut(self, fn):
        try:
            focus = self.root.focus_get()
        except Exception:                                   # tkinter quirk with combobox popups
            focus = None
        if isinstance(focus, (tk.Text, tk.Entry)):
            return                                          # user is typing – don't hijack keys
        fn()

    # ------------------------------------------------------------------ actions
    def handle_events(self, events):
        for kind, val in events:
            if kind == "letter":
                self.flash_until = time.monotonic() + 0.35
            elif kind == "word":
                self.sentence.insert("end", val + " ")
                self.sentence.see("end")
            elif kind == "sentence" and self.auto.get():
                self.translate()

    def word_break(self):
        self.handle_events(self.composer.break_word())

    def backspace(self):
        if self.composer.current_word:
            self.composer.backspace()
            return
        txt = self.sentence.get("1.0", "end-1c").rstrip()
        txt = txt[:txt.rfind(" ") + 1] if " " in txt else ""
        self.sentence.delete("1.0", "end"); self.sentence.insert("1.0", txt)

    def clear(self):
        self.composer.reset()
        self.sentence.delete("1.0", "end")
        self._set_output("")
        self.time_lbl.configure(text="")

    def copy_translation(self):
        if self.last_translation:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_translation)
            self.time_lbl.configure(text="Copied to clipboard")

    def _set_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end"); self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def translate(self):
        self.word_break()                                   # flush the word being spelled
        text = self.sentence.get("1.0", "end-1c").strip()
        if not text or self.translating:
            return
        key = self.key.get().strip()
        if not key:
            self._set_output("Add your Gemini API key in the Settings tab (or set GEMINI_API_KEY).")
            return
        self.translating = True
        self.tr_btn.configure(state="disabled", text="Translating…")
        self.time_lbl.configure(text="")
        self._set_output("")
        args = (text, self.lang.get().strip() or "English", self.model.get().strip(), key)
        threading.Thread(target=self._translate_worker, args=args, daemon=True).start()

    def _translate_worker(self, text, language, model, key):
        t0 = time.perf_counter()
        translation = ""
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=model, contents=PROMPT.format(language=language, text=text),
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   temperature=0.2))
            try:
                d = json.loads(resp.text)
                translation = d.get("translation", "")
                out = f"{d.get('corrected', '')}\n\n{translation}"
                if d.get("romanized"):
                    out += f"\n({d['romanized']})"
            except Exception:
                out = translation = resp.text
        except Exception as e:                              # network, bad key, bad model name…
            out = f"Gemini error: {e}"
        # never touch Tk from this thread: hand the result to the UI loop
        self.results.put(("translation", (out, translation, time.perf_counter() - t0, language)))

    def _translate_done(self, out, translation, seconds, language):
        self._set_output(out)
        self.last_translation = translation
        self.last_language = language
        self.time_lbl.configure(text=f"done in {seconds:.1f} s")
        self.translating = False
        self.tr_btn.configure(state="normal", text="Translate")
        if translation and self.auto_speak.get():
            self.speak()

    # ------------------------------------------------------------------ voice
    def speak(self):
        if self.speaking:                                   # button doubles as "Stop"
            self.stop_speech()
            return
        text = self.last_translation.strip()
        if not text:
            self.time_lbl.configure(text="Translate something first")
            return
        self.speaking = True
        self.speech_token += 1
        self.speak_btn.configure(text="Stop")
        self.time_lbl.configure(text="Preparing voice…")
        threading.Thread(target=self._speak_worker, daemon=True,
                         args=(text, self.last_language, self.voice_menu.get(),
                               self.speech_token)).start()

    def _speak_worker(self, text, language, gender, token):
        try:
            path = synthesize(text, language, gender, self._voice_cache)
            self.results.put(("speech", (path, token)))
        except SpeechError as e:
            self.results.put(("speech_error", (str(e), token)))
        except Exception as e:
            self.results.put(("speech_error", (f"Voice error: {e}", token)))

    def _speech_ready(self, path, token):
        if token != self.speech_token or not self.speaking:  # cancelled while generating
            try:
                os.remove(path)
            except OSError:
                pass
            return
        try:
            self.player.play(path)
            self._playing = True
            self.time_lbl.configure(text="Speaking…")
        except Exception as e:
            self._speech_finished(f"Audio output unavailable: {e}")

    def _speech_finished(self, message=""):
        self.player.stop()                                  # also deletes the temp audio file
        self.speaking = False
        self._playing = False
        self.speak_btn.configure(text="Speak")
        self.time_lbl.configure(text=message)

    def stop_speech(self):
        self.speech_token += 1
        self.player.stop()
        self._speech_finished()

    # ------------------------------------------------------------------ main loop
    def loop(self):
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == "translation":
                    self._translate_done(*payload)
                elif kind == "speech":
                    self._speech_ready(*payload)
                elif kind == "speech_error" and payload[1] == self.speech_token:
                    self._speech_finished(payload[0])
        except queue.Empty:
            pass
        if self._playing and not self.player.busy():
            self._speech_finished()

        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)                      # mirror view; model is mirror-invariant
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
            self.last_ts = ts
            hands = detect_hands(self.landmarker, rgb, ts)

            label, conf = None, 0.0
            if hands:
                proba = self.pipe.predict_proba(hands_to_features(hands)[None])[0]
                i = int(np.argmax(proba))
                label, conf = self.classes[i], float(proba[i])

            now = time.monotonic()
            self.handle_events(self.composer.update(label, conf, bool(hands), now))
            self._draw(frame, hands, label, conf)
            self._refresh(label, conf, bool(hands), now)

        t = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(t - self._t_prev, 1e-3)
        self._t_prev = t
        self._after = self.root.after(10, self.loop)

    def _draw(self, frame, hands, label, conf):
        h, w = frame.shape[:2]
        for hand in hands:
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, tuple(hand[a, :2].astype(int)), tuple(hand[b, :2].astype(int)),
                         (120, 210, 90), 2, cv2.LINE_AA)
            for p in hand:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)

        c = self.composer
        if hands and label:                                  # label chip, top-left
            txt = f"{label}  {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (12, 12), (36 + tw, 28 + th), (32, 26, 20), -1)
            cv2.putText(frame, txt, (24, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif not hands:
            cv2.putText(frame, "Show your hand(s)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (190, 190, 190), 2, cv2.LINE_AA)
        if hands and c.candidate:                            # hold-progress bar, bottom
            color = (110, 200, 60) if c.status == "locked" else (60, 170, 245)
            cv2.rectangle(frame, (20, h - 28), (w - 20, h - 16), (55, 55, 55), -1)
            cv2.rectangle(frame, (20, h - 28), (20 + int((w - 40) * c.progress), h - 16), color, -1)

        frame = cv2.resize(frame, (self.disp_w, self.disp_h))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=img)
        self.video.image = img

    def _refresh(self, label, conf, present, now):
        c = self.composer
        cursor = "▏" if int(now * 2) % 2 == 0 else " "
        self.word_lbl.configure(text=" ".join(c.current_word) + cursor)

        if now - self._pill_at > 0.5:
            self._pill_at = now
            self.pill.configure(text=f"● camera live · {self.fps:.0f} FPS", text_color=OK)

        if not present:
            self.sign_lbl.configure(text="–", text_color=MUTED)
            self.conf_lbl.configure(text="Show your hand")
            self.hold_bar.set(0)
            return
        flashing = now < self.flash_until
        self.sign_lbl.configure(text=label or "?", text_color=OK if flashing else TEXT)
        note = "added · lower hand to repeat" if c.status == "locked" else "hold steady…"
        self.conf_lbl.configure(text=f"{conf * 100:.0f}%  ·  {note}")
        self.hold_bar.configure(progress_color=OK if c.status == "locked" else ACCENT)
        self.hold_bar.set(c.progress)

    def close(self):
        try:
            self.root.after_cancel(self._after)
        except Exception:
            pass
        self.player.stop()
        self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=720, help="camera preview width in pixels")
    ap.add_argument("--conf", type=float, default=0.70, help="min classifier confidence")
    ap.add_argument("--hold", type=float, default=0.8, help="seconds to hold a sign")
    ap.add_argument("--word-gap", type=float, default=1.8, help="seconds of no hands = new word")
    args = ap.parse_args()
    if not CLASSIFIER_PATH.exists():
        raise SystemExit("No trained model found. Run extract_landmarks.py then train.py first.")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
