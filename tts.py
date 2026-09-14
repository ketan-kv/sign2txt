import threading
import pyttsx3

try:
    import pythoncom
except ImportError:
    pythoncom = None

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
                com_initialized = False
                if pythoncom is not None:
                    try:
                        pythoncom.CoInitialize()
                        com_initialized = True
                    except Exception:
                        pass
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.setProperty('volume', 1.0)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[TTS] Speech error: {e}")
                finally:
                    if com_initialized and pythoncom is not None:
                        try:
                            pythoncom.CoUninitialize()
                        except Exception:
                            pass
                    self.is_speaking = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
