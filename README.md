# Sign2Txt: Real-Time Indian Sign Language (ISL) Recognition, Translation & Speech

**Sign2Txt** is an end-to-end assistive desktop application that recognizes Indian Sign Language (ISL) gestures in real-time from a webcam feed, reconstructs continuous fingerspelled gestures into coherent sentences using Generative AI (Gemini 3.6 Flash), translates the text into multiple regional and global languages, and speaks the result aloud with offline Text-to-Speech (TTS).

---

## 🌟 Key Features

### 1. Dual-Mode Gesture Recognition
- **Seamless Peak-Confidence Continuous Streaming (Default)**:
  - Continuously evaluates incoming hand gestures and dynamically locks onto the prediction when confidence reaches a local peak (>60% threshold for ~120ms).
  - Automatically commits characters without requiring pauses or fixed countdown timers.
  - Features zero intrusive screen flash—using a subtle, non-disruptive cyan bounding box pulse when a letter is confirmed.
- **Snapshot Countdown Mode**:
  - A structured 2-second countdown mode designed for training, beginner practice, and precise single-sign validation.

### 2. Robust 132-Dimension Feature Pipeline
- Utilizes Google MediaPipe Hands to capture 21 3D landmarks per hand (up to 2 hands simultaneously).
- Applies **Global Centering**, **Scale Normalization** (bounding box invariant), and **3D Palm Normal Vectors** to capture spatial hand orientation and finger depth.
- Mitigates two-hand occlusion and scale drift across varied camera distances.

### 3. Smart Typing Dynamics
- **Natural Auto-Spacing**: Automatically appends a space when hands drop out of frame after spelling a word.
- **Accurate Backspacing**: Removes the last committed character or trailing whitespace with proper Tkinter indexing and debounce protection.

### 4. AI Typo Repair & Multi-Language Translation
- Powered by Google Gemini (`gemini-3.6-flash`) via the modern `google-genai` SDK.
- Fixes minor fingerspelling typos, phonetically merges letters into proper words, and formats grammatically correct sentences.
- Supports instant translation into:
  - **English**, **Hindi**, **Telugu**, **Tamil**, **Marathi**, **Bengali**, **Spanish**, **French**, and **German**.
  - Built-in automatic fallback to `deep-translator` (Google Translate engine) if API quotas or network constraints occur.

### 5. Asynchronous Speech Synthesis
- Powered by `pyttsx3` for instant, offline voice output.
- Features non-blocking worker threads with Windows COM safety (`pythoncom.CoInitialize()`) to prevent UI freezes and concurrency crashes.

---

## 📁 Project Architecture

```
sign2txt/
├── app.py                  # Main Tkinter GUI application & real-time OpenCV pipeline
├── translator.py           # Gemini 3.6 Flash typo-repair & translation module
├── tts.py                  # Thread-safe pyttsx3 Text-To-Speech engine
├── extractor.py            # 132-D MediaPipe landmark feature extraction utility
├── preprocess.py           # Dataset preprocessing & vector normalization
├── train.py                # Deep Neural Network (DNN) training pipeline
├── isl_dnn_model.keras     # Trained Keras classifier model
├── labels.json             # Sign class index-to-label mapping
├── config.example.json     # Configuration template for Gemini API key
├── requirements.txt        # Python dependency manifest
└── .gitignore              # Ignores credentials, datasets, and local artifacts
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11 recommended.
- A functional webcam.

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ketan-kv/sign2txt.git
   cd sign2txt
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configure API Credentials

Sign2Txt uses Google Gemini for context-aware grammar repair and translation:

1. Copy the example configuration:
   ```bash
   # Windows
   copy config.example.json config.json

   # Linux / macOS
   cp config.example.json config.json
   ```
2. Open `config.json` and insert your Gemini API Key:
   ```json
   {
     "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE"
   }
   ```
> *Note: If no API key is provided, the application will automatically fall back to standard translation.*

---

## 💻 Running the Application

Launch the desktop interface:
```bash
python app.py
```

### Application Controls
| Control | Description |
| :--- | :--- |
| **Continuous / Snapshot Toggle** | Switches between continuous peak streaming and 2s countdown snapshot mode. |
| **Commit Sign / Space** | Manually commits the current predicted sign or inserts a whitespace. |
| **Backspace** | Deletes the last entered character or space. |
| **Clear** | Clears the entire current text buffer. |
| **Language Dropdown** | Selects target language for translation (Hindi, Telugu, Tamil, etc.). |
| **Translate & Speak** | Runs AI typo-correction, translates text, and vocalizes the sentence aloud. |
| **Speak Raw** | Reads out the raw detected fingerspelled string directly. |

---

## 🤝 Contributing & Branching Model

- `master`: Stable production branch with all integrated features.
- `stage-3/translation`: Latest translation and TTS pipeline.
- `stage-2/continuous`: Continuous buffering and dynamic consensus mode.
- `stage-1/detection`: Core hand landmark feature extraction experiments.

To contribute:
1. Create a feature branch (`git checkout -b feature/new-sign-support`).
2. Commit your changes (`git commit -m "Add dynamic gesture support"`).
3. Push to your branch (`git push origin feature/new-sign-support`).
4. Open a Pull Request.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
