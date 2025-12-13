print("Model is initializing, please wait.")

import tkinter as tk
from tkinter import scrolledtext
import threading

import sounddevice as sd
import numpy as np
import torch
import keyboard
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import webrtcvad
import queue
import collections
import time

import pyperclip
import pyautogui
from happytransformer import HappyTextToText, TTSettings

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
PADDING_DURATION_MS = 500
SILENCE_DURATION_MS = 800
VAD_AGGRESSIVENESS = 2 # ignore background noise

print("Loading the model...")

processor = Wav2Vec2Processor.from_pretrained("./results/Wav2Vec2-base-LibriSpeech360h-Custom")
model = Wav2Vec2ForCTC.from_pretrained("./results/Wav2Vec2-base-LibriSpeech360h-Custom")

model.eval()

print("Loading grammar checker...")
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)

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
        self.running = True

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
            while self.running:
                try:
                    frame = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

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
    def stop(self):
        self.running = False

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
    # - "Delete (n) words",  1 < n < 13, 20, 30
    # - "Place dot"
    # - "Place period"
    # - "New paragraph"
    # - "Insert phone number"
    # - "Insert mail"
    # - "Place space"
    # - "Check text"
    # - "Check grammar"
    # - "Stop listening"
    # - "Sleep Typist"
    
###################################################

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30
}

class TypistApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Typist")
        self.root.geometry("400x350")
        self.root.attributes('-topmost', True)

        self.is_awake = False
        self.vad_instance = None
        self.is_running = True

        self.status_label = tk.Label(root, text="Status: SLEEPING 💤", fg="gray", font=("Helvetica", 16, "bold"))
        self.status_label.pack(pady=10)

        self.log_area = scrolledtext.ScrolledText(root, width=45, height=12, font=("Consolas", 9))
        self.log_area.pack(pady=5, padx=10)
        
        self.quit_button = tk.Button(root, text="Exit", command=self.on_close, bg="#ffcccc")
        self.quit_button.pack(pady=5)

        self.thread = threading.Thread(target=self.run_audio_loop, daemon=True)
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def update_status_ui(self, awake):
        if awake:
            self.status_label.config(text="Status: LISTENING 🎙️", fg="green")
        else:
            self.status_label.config(text="Status: SLEEPING 💤", fg="gray")

    def execute_grammar_check(self):
        self.log("[Checking Grammar...]")
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.1)
        
        raw_text = pyperclip.paste()
        if not raw_text:
            self.log("Clipboard is empty.")
            return

        sentences = raw_text.split(". ")

        corrected_sentences = []

        for sentence in sentences:
            if sentence.strip():
                result = happy_tt.generate_text(f"grammar: {sentence}", args=args)
                corrected_sentences.append(result.text)

        corrected_text = " ".join(corrected_sentences)

        if corrected_text != raw_text:
            pyperclip.copy(corrected_text)
            pyautogui.hotkey('ctrl', 'v')
            self.log("[Text corrected]")
        else:
            self.log("[No errors found]")

    def process_command(self, text):
        text = text.strip()

        if not text:
            return False
        
        self.log(f"Typist recognized the following: {text}")

        if "check text" in text or "check grammar" in text:
            content = text.replace("check text", "").strip() if text == "check text" else text.replace("check grammar", "").strip()
            keyboard.write(content)
            self.execute_grammar_check()
        elif "delete last word" in text:
            content = text.replace("delete last word", "").strip()
            keyboard.write(content)
            keyboard.send("ctrl+backspace")
            self.log("[Deleted last word]")
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
                self.log(f"[Action: Deleting last {count} words...]")
                for _ in range(count):
                    keyboard.send("ctrl+backspace")
                    time.sleep(0.05)
                return True
            else:
                self.log("[Error: Could not understand how many words to delete]")
        elif "clear all" in text:
            content = text.replace("clear all", "").strip()
            keyboard.write(content)
            keyboard.send("ctrl+a")
            keyboard.send("backspace")
            self.log("[Cleared all text]")
        elif "enter all" in text or "send all" in text:
            content = text.replace("enter all", "").strip() if text == "enter all" else text.replace("send all", "").strip()
            keyboard.write(content)
            keyboard.send("enter")
            self.log("[Sent text]")
        elif "place dot" in text or "place period" in text:
            content = text.replace("place dot", "").strip() if text == "place dot" else text.replace("place period", "").strip()
            keyboard.write(content)
            keyboard.write(".")
            self.log("[Placed a dot]")
        elif "insert mail" in text:
            content = text.replace("insert mail", "").strip()
            keyboard.write(content)
            keyboard.write(" helloworld@gmail.com ") # can be personalized later
            self.log("[Inserted email]")
        elif "insert phone number" in text:
            content = text.replace("insert phone number", "").strip()
            keyboard.write(content)
            keyboard.write(" (777) 777-7777 ") # can be personalized later
            self.log("[Inserted phone number]")
        elif "new paragraph" in text:
            content = text.replace("new paragraph", "").strip()
            keyboard.write(content)
            keyboard.send("shift+enter")
            self.log("[Started new paragraph]")
        elif "place space" in text:
            content = text.replace("place space", "").strip()
            keyboard.write(content)
            keyboard.write(" ")
            self.log("[Placed space]")
        else:
            keyboard.write(text + " ") # default dictation
            self.log(f"[Typed]: {text}")

    def run_audio_loop(self):
        self.log("Audio recording ready...")
        self.vad_instance = RecordAudio(SAMPLE_RATE, CHUNK_DURATION_MS, PADDING_DURATION_MS, SILENCE_DURATION_MS, VAD_AGGRESSIVENESS)
        
        for audio_chunk in self.vad_instance.generator():
            if not self.is_running: break

            text = transcribe(audio_chunk)

            if not self.is_awake:
                if "wake up typist" in text or "wakeup typist" in text:
                    self.is_awake = True
                    self.update_status_ui(True)
                    self.log("Turned on...")

                    leftover = text.replace("wake up typist", "").replace("wakeup typist", "").strip()
                    if leftover:
                        self.process_command(leftover)
                elif "sleep typist" in text:
                    self.on_close()
                    break
            else:
                if "stop listening" in text:
                    self.is_awake = False
                    self.update_status_ui(False)
                    self.log("Turned off... Say 'Wake up Typist' to continue")
                elif "sleep typist" in text:
                    self.on_close()
                    break
                else:
                    self.process_command(text)
    
    def on_close(self):
        self.log("Shutting down...")
        self.is_running = False
        if self.vad_instance:
            self.vad_instance.stop()
        
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = TypistApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()