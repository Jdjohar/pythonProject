import os
import torch
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig

# ✅ Allow PyTorch 2.6+ Compatibility
torch.serialization.add_safe_globals([XttsConfig])

# ✅ Automatically agree to Coqui TOS
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Load the XTTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)
print(f"✅ Model loaded successfully on {device}")

# ✅ Static path for frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Paths and Constants
CUSTOM_SPEAKER_WAV = "sample.wav"
OUTPUT_PATH = "output.wav"
LANGUAGE = "hi"

# ✅ Ensure speaker WAV file exists
if not os.path.isfile(CUSTOM_SPEAKER_WAV):
    raise FileNotFoundError(f"❌ Speaker WAV not found at {CUSTOM_SPEAKER_WAV}")

# ✅ HTML Home Page
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ✅ TTS Endpoint: Convert text to speech
@app.post("/synthesize/")
async def synthesize(text: str = Form(...)):
    try:
        # Generate speech
        tts.tts_to_file(
            text=text,
            speaker_wav=CUSTOM_SPEAKER_WAV,
            language=LANGUAGE,
            file_path=OUTPUT_PATH,
            sample_rate=48000
        )
        return FileResponse(OUTPUT_PATH, media_type="audio/wav", filename="speech_output.wav")
    except Exception as e:
        return {"error": f"❌ Error during speech synthesis: {str(e)}"}

