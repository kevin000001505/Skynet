import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import logging, torch, os
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import seaborn as sns
import config
from re import sub

class BertPrediction:
    def __init__(
        self,
        model_name: str = "distillbert/distilbert-base-uncased",
        version: str = "0.1"
    ):
        model_dir = f"./BERT/finetuned_models/{sub(r".+/", "", model_name)}_v{version}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def predict(self, text: str):
        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return classifier(text.lower())


class BERTrainer:
    def __init__(
        self,
        use_raw_text: bool = False,
        test_size: float = 0.3,
        model_name: str = "distilbert/distilbert-base-uncased",
        num_labels: int = 2,
        version: str = "0.1",
    ):
        # Load dataset with random forest confidence levels
        logging.info(
            f"Attempt importing training dataset from {config.BIG_DATA_FILE_CLEANED}"
        )
        self.df = pd.read_csv(config.BIG_DATA_FILE_CLEANED)
        logging.info("Training dataset imported successfully")

        self.sanity_check()

        # Initialize variables
        logging.info(f"Filter training dataset by {config.PROBABILITY_THRESHOLD} confidence level")
        self.df = self.df[
            self.df["confidence"] < config.PROBABILITY_THRESHOLD
        ]  # filter by confidence threshold
        logging.info("Structure of filtered dataset:")
        self.df.info(verbose=True)
        logging.info(f"Split train/test sets with {test_size} test set size")
        df_train, df_test = train_test_split(
            self.df, test_size=test_size, shuffle=True, random_state=42
        )

        if use_raw_text:
            logging.info("Using raw text")
            df_train = df_train[["text", "label"]]
            df_test = df_test[["text", "label"]]
        else:
            logging.info("Using pre-processed text")
            df_train = df_train[["preprocess_text", "label"]]
            df_test = df_test[["preprocess_text", "label"]]
            df_train = df_train.rename(columns={"preprocess_text": "text"})
            df_test = df_test.rename(columns={"preprocess_text": "text"})

        logging.info("Structure of train data:")
        df_train.info(verbose=True)
        logging.info("Structure of test data:")
        df_test.info(verbose=True)

        self.model_name = model_name
        self.version = version
        self.finetuned_model_name = f"{sub(r".+/", "", self.model_name)}_v{self.version}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logging.info(f"Model set to version {self.version}")

        # Convert DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(df_train[["text", "label"]])
        test_dataset = Dataset.from_pandas(df_test[["text", "label"]])

        # Tokenize HF datasets, this is the dataset that will be used in training and evaluating
        self.tokenized_train_dataset = train_dataset.map(
            self.tokenize_function, batched=True
        )
        self.tokenized_test_dataset = test_dataset.map(
            self.tokenize_function, batched=True
        )

        # Assign accelerators
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logging.info(f"Accelerator device: {self.device}")

        # Create model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            label2id={0: 0, 1: 1},
            id2label={0: 0, 1: 1},
        )

    def sanity_check(self):
        # Sanity check on dataset
        ## Visual check on dataframe structure
        logging.info("Structure of data:")
        self.df.info(verbose=True)
        logging.info("Value counts of unique values in 'label' column:")
        self.df["label"].value_counts()

        ## Check if required columns exists
        if set(["text", "preprocess_text", "label", "confidence"]).issubset(
            self.df.columns
        ):
            logging.info("All required columns exists")
        else:
            raise ValueError(
                "Dataset must contain columns 'text', 'preprocess_text', 'label', 'confidence'"
            )

        ## Check if label only contain 0 and 1 labels
        labels = self.df["label"].unique()
        if set(self.df["label"].unique() == set([0, 1])):
            logging.info("Label column confirmed to only have 0 and 1 label")
        else:
            raise ValueError(f"Label column unique values are {labels}. Aborting.")

        ## Enforce types
        self.df["preprocess_text"] = self.df["preprocess_text"].astype(str)
        self.df["text"] = self.df["text"].astype(str)
        self.df["label"] = self.df["label"].astype(int)
        self.df["confidence"] = self.df["confidence"].astype(float)

        ## Convert text column to lowercase
        self.df["text"] = self.df["text"].str.lower()

    # Adding short max length to lower training time
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    # compute metrics function
    def compute_metrics(self, p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids

        acc = accuracy_score(labels, preds)
        # Change average to 'weighted' for imbalanced dataset
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    class LossAccuracyLogger(TrainerCallback):
        def __init__(self):
            self.train_loss = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                self.train_loss.append((state.epoch, logs["loss"]))

    def train(
        self,
        learning_rate: float = 1e-5,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: int = 10,
        logging_steps: int = 500,
        save_model_threshold: float = 0.8
    ):

        if not os.path.exists("training_results"):
            logging.debug("training_results folder doesn't exist. Creating one now")
            os.mkdir("training_results")

        training_args = TrainingArguments(
            output_dir="./BERT/training_results/" + self.finetuned_model_name,
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            dataloader_num_workers=5,
            logging_steps=logging_steps,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            bf16=True,
            #torch_compile=True,
            lr_scheduler_type="linear",
        )
        self.logger_callback = self.LossAccuracyLogger()
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[self.logger_callback],
        )
        logging.debug(f"Training arguments: {self.trainer.args}")
        logging.info("Starting training on train set")
        self.trainer.train()

        logging.info("Starting evaluation on test set")
        results = self.trainer.evaluate()
        logging.info(f"Evaluation accuracy: {results['eval_accuracy']}")
        logging.info(f"Evaluation loss: {results['eval_loss']}")

        model_path = f"./BERT/finetuned_models/{self.finetuned_model_name}"
        plot_path = f"./BERT/figures/{self.finetuned_model_name}"
        # Create output dir for models and plots
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        if results["eval_accuracy"] > save_model_threshold:
            logging.info("Model have high enough accuracy. Saving model")
            self.trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)

            # 1. Plot training loss
            train_epochs, train_losses = zip(*self.logger_callback.train_loss)
            plt.figure(figsize=(8, 5))
            plt.plot(train_epochs, train_losses, marker="o")
            plt.title("Training Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(f"{plot_path}/loss_curve.png")
            plt.close()
            logging.info("Saved training loss curve plot")

            # 2. Get predictions on test set
            preds_output = self.trainer.predict(self.tokenized_test_dataset)
            preds = preds_output.predictions.argmax(-1)
            labels = preds_output.label_ids

            # Compute metrics
            acc = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary"
            )

            # 3. Save metrics
            with open(f"{plot_path}/metrics.txt", "w") as f:
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                logging.info("Saved evaluation metrics")

            # 4. Confusion matrix
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(f"{plot_path}/confusion_matrix.png")
            plt.close()
            logging.info("Saved confustion matrix")
