"""Text-to-speech for the translated sentence, using edge-tts (online).

Playback needs no extra Python packages: Windows uses its built-in player, macOS uses afplay,
Linux uses mpg123 / ffplay / mpv if installed.

Voices are looked up at run time from edge-tts's own voice list, so nothing here depends on
hard-coded voice names. If a language has no voice in edge-tts, a clear message is shown.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile

# language name (as shown in the app) -> locale codes to try, in order of preference
LOCALES = {
    "hindi": ["hi-IN"], "english": ["en-IN", "en-US", "en-GB"], "bengali": ["bn-IN", "bn-BD"],
    "tamil": ["ta-IN"], "telugu": ["te-IN"], "kannada": ["kn-IN"], "malayalam": ["ml-IN"],
    "marathi": ["mr-IN"], "gujarati": ["gu-IN"], "punjabi": ["pa-IN"],
    "urdu": ["ur-IN", "ur-PK"], "odia": ["or-IN"], "assamese": ["as-IN"],
    "spanish": ["es-ES", "es-MX"], "french": ["fr-FR"], "german": ["de-DE"],
    "japanese": ["ja-JP"], "chinese": ["zh-CN"], "arabic": ["ar-SA"],
}


class SpeechError(Exception):
    """A problem the user can be told about in plain words."""


def pick_voice(voices: list[dict], language: str, gender: str = "Female") -> str | None:
    """Choose an edge-tts voice ShortName for `language`; prefer `gender`, else any."""
    for locale in LOCALES.get(language.strip().lower(), []):
        cands = [v for v in voices if v.get("Locale", "").lower() == locale.lower()]
        for v in cands:
            if v.get("Gender", "").lower() == gender.lower():
                return v["ShortName"]
        if cands:
            return cands[0]["ShortName"]
    return None


def synthesize(text: str, language: str, gender: str = "Female",
               cache: dict | None = None) -> str:
    """Generate speech for `text` and return the path of a temporary .mp3 file."""
    import edge_tts
    cache = cache if cache is not None else {}

    async def run() -> str:
        if "voices" not in cache:
            cache["voices"] = await edge_tts.list_voices()
        voice = pick_voice(cache["voices"], language, gender)
        if not voice:
            raise SpeechError(f"edge-tts has no voice for {language}.")
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            await edge_tts.Communicate(text, voice).save(path)
        except BaseException:
            os.remove(path)
            raise
        return path

    try:
        return asyncio.run(run())
    except SpeechError:
        raise
    except Exception as e:                                  # no internet, service down, ...
        raise SpeechError(f"Could not create speech (internet needed): {e}")


class _MciBackend:
    """Windows: the built-in MCI audio player (winmm.dll). No extra packages needed."""
    ALIAS = "isl_tts"

    def __init__(self):
        import ctypes
        self._winmm = ctypes.windll.winmm
        self._ctypes = ctypes
        self._open = False

    def _mci(self, command: str) -> str:
        ct = self._ctypes
        buf = ct.create_unicode_buffer(256)
        err = self._winmm.mciSendStringW(command, buf, 255, 0)
        if err:
            msg = ct.create_unicode_buffer(256)
            self._winmm.mciGetErrorStringW(err, msg, 255)
            raise RuntimeError(msg.value or f"MCI error {err}")
        return buf.value

    def start(self, path: str):
        self.stop()
        self._mci(f'open "{path}" type mpegvideo alias {self.ALIAS}')
        self._open = True
        self._mci(f"play {self.ALIAS}")

    def busy(self) -> bool:
        if not self._open:
            return False
        try:
            return self._mci(f"status {self.ALIAS} mode").strip().lower() == "playing"
        except RuntimeError:
            return False

    def stop(self):
        if self._open:
            for command in (f"stop {self.ALIAS}", f"close {self.ALIAS}"):
                try:
                    self._mci(command)
                except RuntimeError:
                    pass
            self._open = False


class _CommandBackend:
    """macOS / Linux: play through a command-line player (afplay, mpg123, ffplay, mpv...)."""

    def __init__(self, cmd: list[str]):
        self._cmd = cmd
        self._proc = None

    def start(self, path: str):
        self.stop()
        self._proc = subprocess.Popen(self._cmd + [path], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    def busy(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


class _PygameBackend:
    """Last-resort fallback if pygame happens to be installed."""

    def __init__(self):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        pygame.mixer.init()
        self._pg = pygame

    def start(self, path: str):
        self._pg.mixer.music.load(path)
        self._pg.mixer.music.play()

    def busy(self) -> bool:
        return bool(self._pg.mixer.music.get_busy())

    def stop(self):
        self._pg.mixer.music.stop()
        if hasattr(self._pg.mixer.music, "unload"):
            self._pg.mixer.music.unload()


_LINUX_PLAYERS = [
    ["mpg123", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpv", "--no-video", "--really-quiet"], ["cvlc", "--play-and-exit", "--quiet"],
]


def _pick_backend():
    if sys.platform == "win32":
        return _MciBackend()
    candidates = [["afplay"]] if sys.platform == "darwin" else _LINUX_PLAYERS
    for cmd in candidates:
        if shutil.which(cmd[0]):
            return _CommandBackend(cmd)
    try:
        return _PygameBackend()
    except Exception:
        raise RuntimeError("No audio player found. Install mpg123, ffmpeg (ffplay) or mpv.")


class Player:
    """Plays one audio file at a time and deletes it afterwards."""

    def __init__(self):
        self._backend = None
        self._path = None

    def play(self, path: str):
        try:
            if self._backend is None:
                self._backend = _pick_backend()
            self._release()
            self._backend.start(path)
            self._path = path
        except Exception:
            _silent_remove(path)
            raise

    def busy(self) -> bool:
        return self._backend is not None and self._backend.busy()

    def stop(self):
        self._release()

    def _release(self):
        if self._backend is not None:
            self._backend.stop()                            # must stop first: Windows locks files
        if self._path:
            _silent_remove(self._path)
            self._path = None


def _silent_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass
