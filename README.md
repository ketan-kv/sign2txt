# ISL Fingerspelling → Words → Translation

Live webcam window: sign Indian Sign Language letters → letters form words → Gemini cleans up
the spelling and translates into the language you pick.

## Setup (once)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Linux only, if tkinter is missing:  sudo apt install python3-tk
```
Kaggle needs credentials for `kagglehub` (a `~/.kaggle/kaggle.json` token, or accept the
dataset terms in the browser first). Or download the dataset manually and pass `--data`.

## Run
```bash
python extract_landmarks.py        # 1) dataset images -> hand landmarks  (10-30 min, one time)
python train.py                    # 2) train classifier                   (a few minutes)
export GEMINI_API_KEY="your-key"   # Windows: set GEMINI_API_KEY=your-key
python app.py                      # 3) live window
```
Quick trial first: `python extract_landmarks.py --max-per-class 200`.

## How to sign in the window
| You do | Result |
|---|---|
| Hold a sign steady ~0.8 s (progress bar fills) | letter added to the current word |
| Change to another sign | next letter (no pause needed) |
| Same letter twice (e.g. LL) | lower your hand out of view for a moment, sign again |
| Lower both hands out of frame ~1.8 s | word ends (space) |
| Stay out of frame ~5 s | sentence ends (auto-translate if the box is ticked) |
| Keyboard: Space / Backspace / Enter | manual word break / delete / translate |

The recognized-text box is editable, so you can fix a letter before translating. Gemini also
corrects likely spelling slips ("HELO" → "hello") before translating.

Tuning: `python app.py --hold 0.6 --word-gap 1.5 --conf 0.6` (shorter hold = faster but more
accidental letters).

## Design notes
* **MediaPipe landmarks + small MLP**, not a CNN on pixels (as in the Gesture-Speech repo).
  Works across backgrounds/lighting and runs in real time on a CPU.
* **Two hands supported.** Many ISL letters use both hands, so features include both hands
  and their relative position. Mirrored copies are added in training, so left/right-handed
  signers and the mirrored preview both work.
* `composer.py` holds all letter/word timing logic (no camera code) – edit it to change behaviour.
* Gemini model name is in the window (or `GEMINI_MODEL` env var). If you get a "model not
  found" error, Google has retired it – type a current Flash model name from
  https://ai.google.dev/gemini-api/docs/models.

## Known limitations
* The dataset only has alphabet images, so this is **fingerspelling**, not full ISL
  (real ISL words/sentences are whole-sign gestures with motion).
* The validation score printed by `train.py` is optimistic (near-duplicate frames). Judge it by
  signing in front of your own camera. If a letter is weak, record a few of your own samples.
* Similar-looking letters (e.g. M/N, U/V) may need good lighting and a clear hand pose.
* Keep only the signer's hands in view.
