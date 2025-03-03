import os
import time
import torch
from pathlib import Path
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig

# ✅ Ensure output directory exists
output_dir = "outputs"
Path(output_dir).mkdir(exist_ok=True)

# ✅ Allowlist for PyTorch 2.6+
torch.serialization.add_safe_globals([XttsConfig])

# ✅ Automatically agree to Coqui license
os.environ["COQUI_TOS_AGREED"] = "1"

# ✅ Initialize XTTS Model
print("⏳ Loading TTS model...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)
print("✅ Model loaded successfully!")

# ✅ Speaker WAV (Reference Voice)
custom_speaker_wav = os.path.join(os.getcwd(), "sample.wav")
if not os.path.isfile(custom_speaker_wav):
    raise FileNotFoundError(f"❌ Speaker WAV not found: {custom_speaker_wav}")

# ✅ Monitor input.txt for new tasks
def watch_for_input():
    input_file = os.path.join(output_dir, "input.txt")
    output_file = os.path.join(output_dir, "output.wav")

    while True:
        if os.path.isfile(input_file):
            print("📝 Detected new text input!")

            try:
                # Read input text
                with open(input_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                if text:
                    # Generate speech (default: Hindi language)
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=custom_speaker_wav,
                        language="hi",
                        file_path=output_file,
                    )

                    print(f"✅ Speech generated: {output_file}")

                    # Cleanup input after processing
                    os.remove(input_file)

            except Exception as e:
                print(f"❌ Error: {e}")

        time.sleep(1)  # Poll every second for new input

if __name__ == "__main__":
    print("👷 Worker started, monitoring input files...")
    watch_for_input()
