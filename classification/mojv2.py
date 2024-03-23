import os

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mistral_checkpoint = "mistralai/Mistral-7B-v0.1"

print(device)
print(f"Using device: {device}")

# Function to compute accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Load and preprocess the AESLC dataset
def load_and_preprocess_dataset():
    dataset = load_dataset("aeslc", split='train')
    emails = [item['email_body'] for item in dataset]
    labels = [1 if item['subject_line'] else 0 for item in dataset]
    return emails, labels

emails, labels = load_and_preprocess_dataset()

# Splitting the dataset into training and validation sets
train_emails, val_emails, train_labels, val_labels = train_test_split(emails, labels, test_size=0.3)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token

train_encodings = tokenizer(train_emails, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_emails, truncation=True, padding=True, max_length=512)

# Custom dataset class
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2)


model.config.pad_token_id = model.config.eos_token_id

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()