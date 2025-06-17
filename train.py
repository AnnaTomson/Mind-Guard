from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("emotion")  # Replace with your own mental health dataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized = dataset.map(tokenize, batched=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

args = TrainingArguments("./results", evaluation_strategy="epoch", per_device_train_batch_size=8)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

trainer.train()

# Save the model weights
torch.save(model.state_dict(), "model_weights.pth")