import os
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse, HTMLResponse
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from starlette.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Allowlist the blocked global (for PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([XttsConfig])

# ✅ Automatically agree to the Coqui license
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

# ✅ Initialize XTTS model
print("⏳ Loading TTS model...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)
print("✅ Model loaded successfully!")

# ✅ Ensure output directory exists
output_dir = "outputs"
Path(output_dir).mkdir(exist_ok=True)

# ✅ Path to the custom speaker reference WAV file
custom_speaker_wav = os.path.join(os.getcwd(), "sample.wav")
if not os.path.isfile(custom_speaker_wav):
    raise FileNotFoundError(f"❌ Custom speaker WAV file not found: {custom_speaker_wav}")

# ✅ HTML Frontend
html_template = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Text-to-Speech (TTS)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>

<body class="bg-light">

    <div class="container mt-5">
        <h1 class="mb-4">Text-to-Speech (TTS) Generator 🎤</h1>

        <form method="post" action="/" enctype="application/x-www-form-urlencoded">
            <div class="mb-3">
                <label for="text" class="form-label">Enter your story:</label>
                <textarea class="form-control" id="text" name="text" rows="5" required>खुशहाल गाँव की कहानी...</textarea>
            </div>

            <button type="submit" class="btn btn-primary">Generate Speech</button>
        </form>

        {% if download_url %}
        <div class="mt-4">
            <a href="{{ download_url }}" class="btn btn-success">Download Speech</a>
        </div>
        {% endif %}
    </div>

</body>

</html>
"""

# ✅ Homepage - Render HTML
@app.get("/", response_class=HTMLResponse)
async def home():
    return html_template.replace("{% if download_url %}...{% endif %}", "")

# ✅ Handle form submission
@app.post("/", response_class=HTMLResponse)
async def generate_speech(request: Request, text: str = Form(...)):
    try:
        # Language (ISO 639-1 code for Hindi is "hi")
        language = "hi"

        # Output file path
        output_path = f"{output_dir}/output.wav"

        # Generate speech
        tts.tts_to_file(text=text, speaker_wav=custom_speaker_wav, language=language, file_path=output_path)

        # Prepare download link
        download_url = f"/download/{os.path.basename(output_path)}"
        return html_template.replace("{% if download_url %}...{% endif %}",
                                     f'<div class="mt-4"><a href="{download_url}" class="btn btn-success">Download Speech</a></div>')

    except Exception as e:
        return HTMLResponse(f"<h3>Error: {e}</h3>")

# ✅ Serve output files for download
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type='audio/wav', filename=filename)
    return HTMLResponse("<h3>File not found!</h3>")

# ✅ Main entry point - Ensure Render detects port
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)