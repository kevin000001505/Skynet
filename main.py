from pipeline.core import run_pipeline_for_dataset, big_data_pipeline
from sklearn.metrics import confusion_matrix
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


<<<<<<< HEAD
def main():
    # List of datasets to process
    datasets = [
        # "Testing",
        "Normal",
        "Twitter",
        "Movie",
        "Yelp",
    ]
=======
def plot_confusion_matrix(result_dict, dataset_name):
    conf_mat = confusion_matrix(result_dict["y_true"], result_dict["y_pred"])
    labels = sorted(list(set(result_dict["y_true"]) | set(result_dict["y_pred"])))
>>>>>>> main

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f'Confusion Matrix for {result_dict["threshold"]} - {dataset_name}, cover {result_dict["coverage_percentage"]:.2f}% coverage'
    )
    plot_filename = (
        f"plots/confusion_matrix_{dataset_name}_{result_dict['threshold']}.png"
    )
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()


def main(big_data: bool = False):
    if big_data:
        dataset_name = "big_data"
        logger.info("Running in big data mode.")
        result_dict = big_data_pipeline(dataset_name)
        if result_dict is None:
            logger.error("Skipping dataset due to result error.")
        if config.DEBUG:
            plot_confusion_matrix(result_dict, dataset_name)

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
    args = parser.parse_args()
    main(big_data=args.bigdata)
