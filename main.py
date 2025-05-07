from pipeline.core import run_pipeline_for_dataset, big_data_pipeline
from sklearn.metrics import confusion_matrix
from transformer.BERT import BERTrainer
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import config
import logging
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(result_dict: dict, model_name: str = None):
    conf_mat = confusion_matrix(
        result_dict["y_true"], result_dict["y_pred"], normalize="all"
    )
    if model_name == "Random Forest":
        conf_mat *= result_dict["coverage_percentage"]

    if model_name == "distilBERT":
        conf_mat *= 1 - result_dict["coverage_percentage"]

    labels = sorted(list(set(result_dict["y_true"]) | set(result_dict["y_pred"])))

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f'Confusion Matrix for {result_dict["threshold"]} - {model_name}, cover {result_dict["coverage_percentage"]:.2f}% coverage'
    )
    plot_filename = (
        f"plots/confusion_matrix_{model_name}_{result_dict['threshold']}.png"
    )
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()


def main(big_data: bool = False, recreate: bool = False, skiprf: bool = False):
    if big_data:
        model_name = "Random Forest"
        logger.info("Running in big data mode.")
        if not skiprf:
            rf_result = big_data_pipeline(model_name, recreate)
            if rf_result is None:
                logger.error("Skipping dataset due to result error.")
            else:
                plot_confusion_matrix(rf_result, model_name)
        trainer = BERTrainer(use_threshold=True, version="1")
        bert_result = trainer.train(
            save_model_threshold=0.8,
            learning_rate=5e-6,
            per_device_train_batch_size=32,
            num_train_epochs=3,
            metric_for_best_model="accuracy",
            eval_steps=1000,
            save_steps=800,
            eval_strategy="epoch",
            save_strategy="best",
            dataloader_num_workers=4,
        )
        plot_confusion_matrix(bert_result, bert_result["model_name"])
        logger.info("BERT model trained and confusion matrix plotted.")

        # Hard Code every stuff
        rf_conf_mat = confusion_matrix(rf_result["y_true"], rf_result["y_pred"])
        bert_conf_mat = confusion_matrix(bert_result["y_true"], bert_result["y_pred"])
        total_conf_mat = rf_conf_mat + bert_conf_mat
        labels = sorted(list(set(rf_result["y_true"]) | set(rf_result["y_pred"])))
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            total_conf_mat,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix for all test data - Hybrid Model")
        plot_filename = "plots/confusion_matrix_hybrid.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(plot_filename)
        plt.close()

    else:
        logger.info("Running in normal mode.")

        for dataset_name in tqdm(
            config.DATA_LOCATIONS.keys(), desc="Processing Datasets"
        ):
            result_dict = run_pipeline_for_dataset(dataset_name)
            if result_dict is None:
                logger.error(f"Skipping dataset due to result error: {dataset_name}")
                continue

            plot_confusion_matrix(result_dict, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline on datasets.")
    parser.add_argument("--bigdata", action="store_true", help="Run on big data mode")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate low confidence data. Using --skiprf will ignore this flag",
    )
    parser.add_argument(
        "--skiprf", action="store_true", help="Skip Random Forest training"
    )
    args = parser.parse_args()
    main(big_data=args.bigdata, recreate=args.overwrite, skiprf=args.skiprf)
