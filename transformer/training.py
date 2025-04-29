import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
from peft import get_peft_model, LoraConfig, TaskType

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

df_train = pd.read_csv(
    "../data/yelp_review_polarity_csv/train.csv", encoding="ISO-8859-1"
)
df_train = df_train[:500]
df_train.dropna(subset=["text", "sentiment"], inplace=True)

# Ensure text data is a list of strings and drop missing values
df_train["text"] = df_train["text"].astype(str)

df_train = df_train.rename(columns={"sentiment": "label"})

df_train["label"] = df_train["label"].astype(int)
print(df_train)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(df_train[["text", "label"]])


# Adding short max length to lower training time
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],  # BERT uses query, key, value in attention
    lora_dropout=0.1,
    bias="all",
    task_type=TaskType.SEQ_CLS,
)

training_args = TrainingArguments(
    output_dir="./results/" + model_name,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True,
    num_train_epochs=3,
    # dataloader_num_workers=16,
    # logging_steps=100,
    weight_decay=0.01,
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True


# Define a compute metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, preds)
    return {"accuracy": accuracy}


from transformers import TrainerCallback


class LossAccuracyLogger(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_accuracy = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_loss.append((state.epoch, logs["loss"]))
            if "eval_accuracy" in logs:
                self.eval_accuracy.append((state.epoch, logs["eval_accuracy"]))


logger_callback = LossAccuracyLogger()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    compute_metrics=compute_metrics,
    callbacks=[logger_callback],
)

# Train the model
trainer.train()

import csv

with open("./logs/training" + model_name + ".csv", "w") as f:
    # Unpack the epoch and values
    train_epochs, train_losses = zip(*logger_callback.train_loss)

    writer = csv.writer(f)
    for i in range(len(train_epochs)):
        writer.writerow(train_epochs[i], train_losses[i])
