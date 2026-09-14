import json
import os
from google import genai
from deep_translator import GoogleTranslator

class SignTranslator:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.client = None
        self.model_name = "gemini-3.6-flash"
        self._init_client()

    def _init_client(self):
        """Loads API key from config.json if available."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    api_key = cfg.get("gemini_api_key", "").strip()
                    if api_key:
                        self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"[Translator] Failed to load config: {e}")

    def translate(self, raw_text, target_language="Hindi"):
        """
        Translates raw sign-language text.
        1. Corrects typos and missing spaces.
        2. Translates into target language.
        Returns a dict: {'corrected_english': str, 'translated_text': str, 'engine': str}
        """
        raw_text = raw_text.strip()
        if not raw_text:
            return {"corrected_english": "", "translated_text": "", "engine": "none"}

        # Attempt 1: Gemini AI (Context-aware typo correction + translation)
        if self.client:
            try:
                prompt = (
                    f"You are an expert AI assistant embedded in a Sign Language Detection application.\n"
                    f"The user has signed the following raw text from webcam gestures: \"{raw_text}\"\n\n"
                    f"Tasks:\n"
                    f"1. Fix any spelling typos, accidental double letters, or missing spaces into natural, clear English.\n"
                    f"2. Translate that natural English into: {target_language}.\n\n"
                    f"Respond strictly in this JSON format without markdown code fences:\n"
                    f'{{"corrected_english": "...", "translated_text": "..."}}'
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                resp_text = response.text.strip()
                # Clean any markdown code blocks if present
                if resp_text.startswith("```"):
                    lines = resp_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    resp_text = "\n".join(lines).strip()

                data = json.loads(resp_text)
                return {
                    "corrected_english": data.get("corrected_english", raw_text),
                    "translated_text": data.get("translated_text", ""),
                    "engine": "gemini"
                }
            except Exception as e:
                print(f"[Translator] Gemini API error: {e}. Falling back to deep-translator.")

        # Attempt 2: Fallback Translator (Direct translation without typo repair)
        try:
            # Map language name to lowercase code for deep_translator
            lang_map = {
                "hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn",
                "malayalam": "ml", "bengali": "bn", "marathi": "mr", "gujarati": "gu",
                "spanish": "es", "french": "fr", "german": "de", "japanese": "ja"
            }
            lang_code = lang_map.get(target_language.lower(), target_language.lower())
            translated = GoogleTranslator(source='auto', target=lang_code).translate(raw_text)
            return {
                "corrected_english": raw_text,
                "translated_text": translated,
                "engine": "fallback"
            }
        except Exception as e:
            print(f"[Translator] Fallback translator error: {e}")
            return {
                "corrected_english": raw_text,
                "translated_text": f"[Translation Error: {e}]",
                "engine": "error"
            }
