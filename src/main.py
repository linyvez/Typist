from torchaudio.datasets import LIBRISPEECH
from pathlib import Path
import torchaudio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments,
    logging
)
from torch.utils.data import Dataset
import evaluate
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

# loading dataset
root = Path("data/raw/LIBRISPEECH")
root.mkdir(parents=True, exist_ok=True)

train_dataset = LIBRISPEECH(root=root, url="train-clean-100", download=True)
eval_dataset = LIBRISPEECH(root=root, url="dev-clean", download=True)

# tokenizing transcripts
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

class LibriSpeechDataset(Dataset):
    def __init__(self, torchaudio_dataset, tokenizer):
        self.dataset = torchaudio_dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if len(data) == 2:
            waveform, sr = data
            transcript = ""
        else:
            waveform, sr, transcript, *_ = data

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        input_values = waveform.squeeze(0).numpy()
        labels = self.tokenizer(transcript).input_ids
        return {"input_values": input_values, "labels": labels}

    def __len__(self):
        return len(self.dataset)


train_dataset = LibriSpeechDataset(train_dataset, tokenizer)
eval_dataset = LibriSpeechDataset(eval_dataset, tokenizer)

# initializing wav2vec2 model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
model.freeze_feature_encoder()

def data_collator(batch):
    input_values = [torch.tensor(b["input_values"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]

    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_values": input_values, "labels": labels}

# training, fine-tuning
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)

    # decode
    pred_str = tokenizer.batch_decode(pred_ids)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids)

    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str)
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_strategy="steps",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    max_grad_norm=1.0,
    gradient_accumulation_steps=2,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# metrics
eval_results = trainer.evaluate()
print(f"Validation WER: {eval_results['eval_wer']}")
print(f"Validation CER: {eval_results['eval_cer']}")

test_dataset = LIBRISPEECH(root=root, url="test-clean", download=True)
test_dataset = LibriSpeechDataset(test_dataset)

test_results = trainer.evaluate(test_dataset)
print(f"Test WER: {test_results['eval_wer']:.4f}")
print(f"Test CER: {test_results['eval_cer']:.4f}")
