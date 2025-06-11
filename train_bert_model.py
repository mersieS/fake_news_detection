import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch

df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

df_fake['label'] = 0
df_real['label'] = 1

df = pd.concat([df_fake[['text', 'label']], df_real[['text', 'label']]]).sample(frac=1, random_state=42)
df = df.reset_index(drop=True)

dataset = Dataset.from_pandas(df)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

train_test = dataset.train_test_split(test_size=0.2)
train_ds = train_test['train']
eval_ds = train_test['test']

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./bert_model',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

trainer.train()

model.save_pretrained('./bert_model')
tokenizer.save_pretrained('./bert_model')
print("Model ve tokenizer 'bert_model/' klasörüne kaydedildi.")
