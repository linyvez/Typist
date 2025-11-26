print("Model is initializing, please wait.")

import sounddevice as sd
import numpy as np
import torch
import keyboard
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import webrtcvad
import queue
import collections
import sys
import time

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
PADDING_DURATION_MS = 500
SILENCE_DURATION_MS = 800
VAD_AGGRESSIVENESS = 2 # ignore background noise

processor = Wav2Vec2Processor.from_pretrained("./results/Wav2Vec2-base-LibriSpeech100h-Custom")
model = Wav2Vec2ForCTC.from_pretrained("./results/Wav2Vec2-base-LibriSpeech100h-Custom")

model.eval()

class RecordAudio:
    def __init__(self, sample_rate, chunk_duration_ms, padding_ms, silence_duration_ms, vad_level):
        self.sample_rate = sample_rate
        self.frame_duration_ms = chunk_duration_ms
        self.frame_size = int(sample_rate * chunk_duration_ms / 1000)
        self.vad = webrtcvad.Vad(vad_level)
        self.queue = queue.Queue()
        maxlen = int(padding_ms / chunk_duration_ms)
        self.ring_buffer = collections.deque(maxlen=maxlen)
        self.triggered = False
        self.num_voiced = 0
        self.num_unvoiced = 0
        self.silence_threshold = int(silence_duration_ms / chunk_duration_ms)

    def audio_callback(self, indata, frames, time, status):
        if status: 
            print(status) # error

        data = (indata * 32767).astype(np.int16).flatten()

        for i in range(0, len(data), self.frame_size):
            frame = data[i:i+self.frame_size]

            if len(frame) == self.frame_size:
                self.queue.put(frame)
    
    def generator(self):
        voiced_frames = []

        with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sample_rate, blocksize=self.frame_size):
            while True:
                frame = self.queue.get()
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

                if not self.triggered:
                    self.ring_buffer.append(frame)
                    if is_speech: 
                        self.num_voiced += 1
                    else: 
                        self.num_voiced = 0

                    if self.num_voiced > 5: # speech detected
                        self.triggered = True
                        voiced_frames.extend(self.ring_buffer)
                        self.ring_buffer.clear()
                        self.num_voiced = 0
                else:
                    voiced_frames.append(frame)
                    if not is_speech:
                        self.num_unvoiced += 1
                    else:
                        self.num_unvoiced = 0

                    if self.num_unvoiced > self.silence_threshold: # silence - stop recording
                        self.triggered = False

                        if voiced_frames:
                            full_audio = np.concatenate(voiced_frames).astype(np.float32) / 32768.0
                            yield full_audio

                        voiced_frames = []
                        self.num_unvoiced = 0
                        self.ring_buffer.clear()

def transcribe(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    preds = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(preds)[0]

    return text.lower()

################ LIST OF COMMANDS #################

    # - "Wake up Typist"
    # - "Clear all"
    # - "Send all" 
    # - "Enter all" 
    # - "Delete last word" 
    # - "Delete last sentence" - in progress...
    # - "Delete (n) words" - currently can be unstable
    # - "Place dot"
    # - "Place period"
    # - "New paragraph"
    # - "Insert phone number"
    # - "Insert mail"
    # - "Place space"
    # - "Stop listening"
    # - "Sleep Typist"
    
###################################################

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30
}

def process_command(text):
    text = text.strip()

    if not text:
        return False
    print("Typist recognized the following:", text)

    if "delete last word" in text:
        keyboard.send("ctrl+backspace")
        print("[Deleted last word]")
    elif "delete" in text and "words" in text:
        count = 0
        words_in_text = text.split()
        
        for word in words_in_text:
            if word in WORD_TO_NUM:
                count = WORD_TO_NUM[word]
                break
            
            if word.isdigit():
                count = int(word)
                break

        if count > 0:
            print(f"[Action: Deleting last {count} words...]")
            for _ in range(count):
                keyboard.send("ctrl+backspace")
                time.sleep(0.05)
            return True
        else:
            print("[Error: Could not understand how many words to delete]")
    elif "clear all" in text:
        keyboard.send("ctrl+a")
        keyboard.send("backspace")
        print("[Cleared all text]")
    elif "enter all" in text or "send all" in text:
        keyboard.send("enter")
        print("[Sent text]")
    elif "place dot" in text or "place period" in text:
        keyboard.write(".")
        print("[Placed a dot]")
    elif "insert mail" in text:
        keyboard.write("helloworld@gmail.com") # can be personalized later
        print("[Inserted email]")
    elif "insert phone number" in text:
        keyboard.write("(777) 777-7777") # can be personalized later
        print("[Inserted phone number]")
    elif "new paragraph" in text:
        keyboard.send("shift+enter")
        print("[Started new paragraph]")
    elif "place space" in text:
        keyboard.write(" ")
        print("[Placed space]")
    else:
        keyboard.write(text + " ") # default dictation
        print(f"[Typed]: {text}")

    return True

def main():
    print("\n" + "="*50)
    print("Typist started. Say 'Wake up Typist' to start.")
    print("="*50 + "\n")

    vad = RecordAudio(SAMPLE_RATE, CHUNK_DURATION_MS, PADDING_DURATION_MS, SILENCE_DURATION_MS, VAD_AGGRESSIVENESS)
    is_awake = False

    for audio_chunk in vad.generator():
        text = transcribe(audio_chunk)

        if not is_awake:
            if "wake up typist" in text or "wakeup typist" in text:
                is_awake = True
                print("\nTurned on...")

                leftover = text.replace("wake up typist", "").replace("wakeup typist", "").strip()
                if leftover:
                    process_command(leftover)
            elif "sleep typist" in text:
                print("\n" + "="*50)
                print("Shutting down...")
                print("="*50 + "\n")
                sys.exit(0)
        else:
            if "stop listening" in text:
                is_awake = False
                print("\nTurned off... Say 'Wake up Typist' to continue")
            elif "sleep typist" in text:
                print("\n" + "="*50)
                print("Shutting down...")
                print("="*50 + "\n")
                sys.exit(0)
            else:
                process_command(text)

if __name__ == "__main__":
    main()
