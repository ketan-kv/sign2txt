import threading
import pyttsx3

class SpeechEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self.is_speaking = False

    def speak(self, text):
        """Speaks the text asynchronously in a background thread without blocking the GUI."""
        if not text or not text.strip():
            return
            
        def _worker():
            with self._lock:
                self.is_speaking = True
                try:
                    # Initialize local engine instance per thread for stability on Windows SAPI5
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)     # Natural speaking pace
                    engine.setProperty('volume', 1.0)   # Full volume
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[TTS] Speech error: {e}")
                finally:
                    self.is_speaking = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
