# Typist

## Authors:
- Viktoriia Lushpak (https://github.com/linyvez)
- Oleksandra Shergina (https://github.com/shshrg)

## About
Typist is a speech-to-text AI assistant that helps with typing and editing text in real time. It focuses on making voice typing easier, and also improves text input by adding various commands, such as "insert", "delete", "check grammar" and many others. To start using Typist, you need to wake it up using the voice command "wakeup Typist". At this stage, the assistant only supports English. In the future, support for other languages, personalization (working with personal data, template support), and interface improvements are also planned.

Typist is based on a pre-trained wav2vec2-base model, which was trained on the LibriSpeech dataset, as well as our custom dataset for domain adaptation. To create a syntetic dataset with commands and wakeup word, we used TTS library together with voices from LibriSpeech's recordings. The inference is supported by VAD, which allows to process the dictation and commands at the same time without losing any information.

## Commands
Currently, the Typist supports the following commands:
- "Wake up Typist" - start processing text
- "Clear all"
- "Send all" 
- "Enter all" 
- "Delete last word" 
- "Delete (n) words",  1 < n < 13, 20, 30
- "Place dot"
- "Place period"
- "New paragraph"
- "Insert phone number" - currently inserts a placeholder
- "Insert mail" - currently inserts a placeholder
- "Place space"
- "Check text"
- "Check grammar"
- "Stop listening" - stop processing text
- "Sleep Typist" - close the program

## Prerequisites
All main prerequisites can be found in requirements.txt:

```
pip install -r requirements.txt
```

## How to build
This repository does not include already fine-tuned model. However, you can train the model yourself following these rules:

1. Make sure you have all prerequisites.
2. Run training script, which is available in .py and .ipynb formats. Full GPU training on 100 hours of LibriSpeech for 5 epochs takes approximately 10 hours; training on 360 hours takes approximately 20-30 hours.
3. Generate syntetic datasets for commands and wakeup word using data_generator.ipynb.
4. Continue training with domain adaptation.
5. Run the main.py script. Make sure to put the loaded model inside of the results/ folder.

The successful result of running a final script is a small window indicating Typist's workflow.

## Results
Typist uses small window to inform users with its workflow. To start working with it, say "Wake up Typist" and processing will start. You can dictate usual text, as well as edit it with special commands. To stop processing text but still be enabled, say "Stop listening". To close the program and stop recording entirely, say "Sleep Typist".

Overall, Typist achieved WER of 0.15 and CER of 0.035. However, these results may not be accurate since LibriSpeech contained some corrupted transcripts (e.g., "midle" vs "middle"). Nevertheless, Typist managed to correct such samples, which proves that its real WER score is approximately 0.05-0.1. For comparison, OpenAI Whisper has WER of approximately 0.027-0.04, Amazon Transcribe - 0.065-0.075 and Google Cloud STT - 0.05-0.06, which shows that Typist approaches the performance of commertial engines while maintaining unique text editing features.

## Notes and Tips
The final model was trained on 360 hours of LibriSpeech dataset to understand general english, which means Typist can still make mistakes. Here are general tips for its usage:
- Try to avoid small dictations. Typist can understand well large sentences as it was trained on audiobooks, but small batches, especially with only one word, can produce bad results.
- Dictate the text not too slow, but not too fast.
- Try to pronounce the words clearly, especially without 'eating' the endings.
- Try to avoid noisy background. However, you can calibrate VAD's aggressiveness in the main script.
- Most of the mistakes can be fixed with grammar checker, but this is a basic tool, which was not adapted to Typist's purposes, so be careful with it.
- Feel free to use many commands in one dictation - Typist will process all of them one by one without losing any information!