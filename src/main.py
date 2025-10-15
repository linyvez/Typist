from torchaudio.datasets import LIBRISPEECH
from pathlib import Path
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, logging
from torch.utils.data import Dataset, DataLoader
import evaluate
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

# loading dataset
root = Path("data/raw/LIBRISPEECH")
train_dataset = LIBRISPEECH(root=root, url="train-clean-100", download=True)
eval_dataset = LIBRISPEECH(root=root, url="dev-clean", download=True)

# tokenizing transcripts
class LibriSpeechDataset(Dataset):
    def __init__(self, torchaudio_dataset):
        self.dataset = torchaudio_dataset

    def __getitem__(self, idx):
        waveform, sr, transcript, _, _, _ = self.dataset[idx]
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        input_values = waveform.squeeze(0)
        labels = transcript
        return {"input_values": input_values, "labels": labels}

    def __len__(self):
        return len(self.dataset)

train_dataset = LibriSpeechDataset(train_dataset)
eval_dataset = LibriSpeechDataset(eval_dataset)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# initializing wav2vec2 model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

def data_collator(batch):
    audio_features = [item["input_values"].numpy() if isinstance(item["input_values"], torch.Tensor) else item["input_values"] for item in batch]
    transcripts = [item["labels"] for item in batch]

    batch = processor(
        audio=audio_features,
        text=transcripts,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    return batch

# training, fine-tuning

# loaders for manual fine-tuning:
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
# eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(logits), dim=-1)

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    save_steps=500,
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    warmup_steps=500,
    logging_dir="./logs",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# metrics
eval_results = trainer.evaluate()
print(f"WER: {eval_results['eval_wer']}")
print(f"CER: {eval_results['eval_cer']}")
