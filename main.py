from pipeline.core import run_pipeline_for_dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # List of datasets to process
    datasets = [
        "Testing",
        "Twitter",
        "Movie",
        "Normal",
        "Yelp",
    ]

    # Run the pipeline for each dataset
    for dataset_name in tqdm(datasets, desc="Processing Datasets"):
        # return dictionary with y_true and y_pred
        result_dict = run_pipeline_for_dataset(dataset_name, config)
        if result_dict is None:
            logger.error(f"Skipping dataset due to result error: {dataset_name}")
            continue
        # conf_matrix = confusion_matrix(result_dict["y_true"], result_dict["y_pred"])
        # print("Confusion Matrix:")
        # print(conf_matrix)
        ConfusionMatrixDisplay.from_predictions(
            result_dict["y_true"], result_dict["y_pred"]
        )
        plt.title(f"Confusion Matrix for {result_dict['threshold']} - {dataset_name}")
        plt.show()
        breakpoint()


if __name__ == "__main__":
    main()
