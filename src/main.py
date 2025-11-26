import sounddevice as sd
import numpy as np
import torch
import keyboard
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

print("Model is initializing, please wait.")

processor = Wav2Vec2Processor.from_pretrained("./results/Wav2Vec2-base-LibriSpeech100h")
model = Wav2Vec2ForCTC.from_pretrained("./results/Wav2Vec2-base-LibriSpeech100h")

model.eval()

SAMPLE_RATE = 16000
CHUNK_DURATION = 2

def record_audio():
    print("Typist is listening...")

    audio = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

    audio = audio.flatten()
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    return audio

def transcribe(audio):
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    preds = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(preds)[0]

    return text.lower()

def process_command(text):
    print("Typist recognized the following:", text)

    if "delete last word" in text:
        keyboard.send("ctrl+backspace")
        print("Deleted last word (or at least tried to).")
    elif "clear all" in text:
        keyboard.send("ctrl+a")
        keyboard.send("backspace")
        print("Cleared all text (or at least tried to).")
    elif "enter" in text:
        content = text.replace("enter", "").strip()
        keyboard.write(content)
        keyboard.send("enter")
        print("Sent text")
    else:
        keyboard.write(text + " ") # default dictation

def is_silence(audio, threshold=0.1):
    rms = np.sqrt(np.mean(audio ** 2))
    print(rms)
    return rms < threshold

def main():
    print("Typist started. Press CTRL+C to stop.")

    while True:
        audio = record_audio()

        if is_silence(audio):
            print("Skip silence...")
            continue

        text = transcribe(audio)
        process_command(text)

if __name__ == "__main__":
    main()
