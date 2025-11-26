from torchaudio.datasets import LIBRISPEECH
from pathlib import Path
import torchaudio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
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

train_ds = LIBRISPEECH(root=root, url="train-clean-100", download=True)
eval_ds = LIBRISPEECH(root=root, url="dev-clean", download=True)

# tokenizing transcripts
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = processor.tokenizer
feature_extractor = processor.feature_extractor

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

        return {"input_values": input_values, "labels": transcript}

    def __len__(self):
        return len(self.dataset)


train_dataset = LibriSpeechDataset(train_ds, tokenizer)
eval_dataset = LibriSpeechDataset(eval_ds, tokenizer)

# initializing wav2vec2 model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base",
                                       pad_token_id=processor.tokenizer.pad_token_id,
                                       vocab_size=len(processor.tokenizer),
                                       ctc_loss_reduction="mean"
                                       )
model.freeze_feature_encoder()

def data_collator(batch):
    audio = [b["input_values"] for b in batch]
    text = [b["labels"] for b in batch]

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    labels_batch = tokenizer(
        text,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    labels = labels_batch.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

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

    print("\n" + "="*30)
    print(f"SAMPLE 1 TARGET: {label_str[0]}")
    print(f"SAMPLE 1 PRED:   {pred_str[0]}")
    print("-" * 10)
    print(f"SAMPLE 2 TARGET: {label_str[1]}")
    print(f"SAMPLE 2 PRED:   {pred_str[1]}")
    print("="*30 + "\n")

    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str)
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    warmup_steps=1000,
    logging_steps=100,
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
print(f"Validation WER: {eval_results['eval_wer']:.4f}")
print(f"Validation CER: {eval_results['eval_cer']:.4f}")

test_dataset = LIBRISPEECH(root=root, url="test-clean", download=True)
test_dataset = LibriSpeechDataset(test_dataset, tokenizer)

test_results = trainer.evaluate(test_dataset)
print(f"Test WER: {test_results['eval_wer']:.4f}")
print(f"Test CER: {test_results['eval_cer']:.4f}")
