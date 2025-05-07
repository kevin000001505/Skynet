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
from sklearn.model_selection import StratifiedKFold
import logging, torch, os
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import seaborn as sns
import config
from shutil import rmtree


class BertPrediction:
    def __init__(
        self,
        version: str = "0.1",
    ):
        # Use the fold with best accuracy
        max_accuracy = -1
        fold = 1
        for i, folder in enumerate(os.listdir(f"./BERT/figures/model_v{version}")):
            with open(
                f"./BERT/figures/model_v{version}/{folder}/metrics_{folder}.txt", "r"
            ) as f:
                text = f.readline()
                accuracy = float(text.split()[1])
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    fold = i
        if max_accuracy == -1:
            raise FileNotFoundError(
                f"No valid metrics folder found. Please check your BERT/figures/model_v{version} folder"
            )
        logging.info(
            f"Using best distilBERT model at fold {fold} with accuracy {max_accuracy}."
        )

        model_dir = f"./BERT/finetuned_models/model_v{version}/fold_{fold}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased",
            padding="max_length",
            truncation=True,
            max_length=config.TOKENIZER_MAX_LENGTH,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )

    def predict(self, text: str):
        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            padding="max_length",
            truncation=True,
            max_length=config.TOKENIZER_MAX_LENGTH,
        )
        return classifier(text)


class BERTrainer:
    def __init__(
        self,
        k: int = 5,
        model_name: str = "distilbert/distilbert-base-uncased",
        num_labels: int = 2,
        use_threshold: bool = False,
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
        if use_threshold:
            logging.info(
                f"Filter training dataset by {config.PROBABILITY_THRESHOLD} confidence level"
            )
            self.df = self.df[
                self.df["confidence"] < config.PROBABILITY_THRESHOLD
            ]  # filter by confidence threshold
        else:
            logging.info("Use all dataset for training")

        if use_threshold:
            logging.info("Structure of filtered dataset:")
            self.df.info(verbose=True)
        self.df = self.df[["text", "label"]]  # Keep only raw text and label

        self.model_name = model_name
        self.version = version
        self.finetuned_model_name = f"model_v{self.version}"
        logging.info(f"Model set to version {self.version}")

        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Convert DataFrames to Hugging Face Datasets
        dataset = Dataset.from_pandas(self.df[["text", "label"]])

        # Tokenize HF datasets, this is the dataset that will be used in training and evaluating
        self.tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

        # Initialize K-Fold for cross validation
        logging.info(f"Initialize cross validation using {k} folds")
        self.kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # Assign accelerators
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logging.info(f"Accelerator device: {self.device}")

        self.num_labels = num_labels

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
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.TOKENIZER_MAX_LENGTH,
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
        dataloader_num_workers: int = 4,  # Change according the the amount of system ram
        learning_rate: float = 5e-6,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: int = 4,
        logging_steps: int = 1000,
        save_model_threshold: float = 0.8,
        metric_for_best_model="accuracy",
        eval_steps: int = 500,
        save_steps: int = 1000,
        eval_strategy: str = "steps",
        save_strategy: str = "steps",
    ):
        # Clean up space before training
        rmtree("BERT/training_results", ignore_errors=True)

        max_acc = -1  # metric to return best confusion matrix
        ret = None
        # Begin training
        for fold, (train_idx, val_idx) in enumerate(
            self.kf.split(self.df, self.df["label"])
        ):
            logging.info(f"Fold {fold + 1}")

            training_args = TrainingArguments(
                output_dir=f"./BERT/training_results/{self.finetuned_model_name}/fold_{fold + 1}",
                overwrite_output_dir=True,
                learning_rate=learning_rate,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                dataloader_num_workers=dataloader_num_workers,
                logging_steps=logging_steps,
                optim="adamw_torch",
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                lr_scheduler_type="linear",
                seed=42,
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
                save_strategy=save_strategy,
                save_steps=save_steps,
                save_total_limit=1,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=True,
                load_best_model_at_end=True,
            )

            # Split data into train/test sets
            tokenized_train_dataset = self.tokenized_dataset.select(train_idx)
            tokenized_test_dataset = self.tokenized_dataset.select(val_idx)

            # Create model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                label2id={0: 0, 1: 1},
                id2label={0: 0, 1: 1},
            )

            self.logger_callback = self.LossAccuracyLogger()
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_test_dataset,
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

            model_path = (
                f"./BERT/finetuned_models/{self.finetuned_model_name}/fold_{fold + 1}"
            )
            plot_path = f"./BERT/figures/{self.finetuned_model_name}/fold_{fold + 1}"
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
            plt.title(f"Training Loss per Epoch for distilBERT at fold {fold + 1}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(f"{plot_path}/loss_curve_fold_{fold + 1}.png")
            plt.close()
            logging.info("Saved training loss curve plot")

            # 2. Get predictions on test set
            preds_output = self.trainer.predict(tokenized_test_dataset)
            preds = preds_output.predictions.argmax(-1)
            labels = preds_output.label_ids

            # Compute metrics
            acc = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary"
            )

            if acc > max_acc:
                max_acc = acc
                ret = {"y_true": labels, "y_pred": preds, "dataset_name": "distilBERT"}

            # 3. Confusion matrix
            cm = confusion_matrix(labels, preds, normalize="all") * 100

            # 4. Save metrics
            with open(f"{plot_path}/metrics_fold_{fold + 1}.txt", "w") as f:
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Confusion matrix:\n{cm}")
                logging.info("Saved evaluation metrics")

            # 5. Save confusion matrix as plot
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix for distilBERT at fold {fold + 1}")
            plt.tight_layout()
            plt.savefig(f"{plot_path}/confusion_matrix_fold_{fold + 1}.png")
            plt.close()
            logging.info("Saved confustion matrix")

        # Clean up space after training
        rmtree("BERT/training_results", ignore_errors=True)
        return ret
