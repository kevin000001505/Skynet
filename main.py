from pipeline.core import run_pipeline_for_dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
import logging
import seaborn as sns
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # List of datasets to process
    datasets = [
        # "Testing",
        "Normal",
        "Twitter",
        "Movie",
        "Yelp",
    ]

    for dataset_name in tqdm(datasets, desc="Processing Datasets"):
        # return dictionary with y_true, y_pred and threshold
        result_dict = run_pipeline_for_dataset(dataset_name, config)
        if result_dict is None:
            logger.error(f"Skipping dataset due to result error: {dataset_name}")
            continue

        conf_mat = confusion_matrix(result_dict["y_true"], result_dict["y_pred"])
        labels = sorted(list(set(result_dict["y_true"]) | set(result_dict["y_pred"])))

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
        plt.title(f'Confusion Matrix for {result_dict["threshold"]} - {dataset_name}')
        plot_filename = (
            f"plots/confusion_matrix_{dataset_name}_{result_dict['threshold']}.png"
        )
        plt.savefig(plot_filename)
        plt.close()


if __name__ == "__main__":
    main()
