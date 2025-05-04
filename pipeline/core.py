from data_processing.loader import load_csv_data, load_big_data_csv
from utils.helpers import split_data
from modeling.random_forest import RandomForestModel
import logging, config
import os


logger = logging.getLogger(__name__)


def run_pipeline_for_dataset(dataset_name: str):
    """Runs the full pipeline for a single dataset."""

    logger.info(f"Starting pipeline for dataset: {dataset_name}")

    # 1. Load Data (using the function from loader.py)
    raw_df = load_csv_data(dataset_name)

    # Check if loading was successful
    if raw_df is None:
        logger.warning(f"Skipping dataset due to loading error: {dataset_name}")
        return  # Stop processing this dataset

    # 2. Clean Data
    cleaner_class = config.CLEANING_STRATEGIES.get(
        dataset_name, config.CLEANING_STRATEGIES.get("default")
    )
    if not cleaner_class:
        logger.error(f"No cleaner found for dataset: {dataset_name}. Skipping.")
        return

    cleaner = cleaner_class()
    cleaned_df = cleaner.clean(
        raw_df.copy()
    )  # Pass a copy to avoid modifying original df unintentionally
    logger.info(
        f"Data cleaned using {cleaner_class.__name__}. Shape: {cleaned_df.shape}"
    )
    target_column = config.TARGET_COLUMNS.get(dataset_name)

    if target_column not in cleaned_df.columns:
        logger.error(
            f"Target column '{target_column}' not found in cleaned data for {dataset_name}. Skipping."
        )
        return
    x_train, x_test, y_train, y_test, le = split_data(cleaned_df, target_column)

    # 4. Train Model
    model = RandomForestModel(params=config.RANDOM_FOREST_PARAMS)
    model.train(x_train, y_train)
    logger.info("Random Forest model trained.")

    # 5. Predict Probabilities
    probabilities = model.predict_proba(x_test)
    logger.info("Probabilities predicted.")

    filtered_output = model.filter_by_threshold(
        y_pred=probabilities,
        y_true=y_test,
        threshold=config.PROBABILITY_THRESHOLD,
        labels_encoder=le,
    )
    model.save_low_confidence_to_csv(
        x=cleaned_df["preprocess_text"],
        threshold=config.PROBABILITY_THRESHOLD,
        dataset_name=dataset_name,
        cleaned_data=cleaned_df,
    )

    logger.info("Low confidence samples saved.")

    # 6. Save Model
    models_dir = os.path.join(config.PROJECT_ROOT, "models")
    model.save_model(models_dir, dataset_name)

    return filtered_output


def big_data_pipeline(dataset_name: str, recreate: bool = False):
    """Runs the pipeline for big data."""
    big_data = load_big_data_csv(recreate)

    # If data is already cleaned skip cleaning data
    if os.path.exists(config.BIG_DATA_FILE_CLEANED) and not recreate:
        cleaned_df = big_data
    else:
        cleaner = config.CLEANING_STRATEGIES.get("default")()
        cleaned_df = cleaner.big_data_clean(big_data.copy())

    print("Unique Labels: ", cleaned_df["label"].unique())
    logger.info(f"Big Data Shape: {cleaned_df.shape}")
    target_column = config.TARGET_COLUMNS.get("default")

    x_train, x_test, y_train, y_test, le = split_data(cleaned_df, target_column)

    # 4. Train Model
    model = RandomForestModel(params=config.RANDOM_FOREST_PARAMS)
    model.train(x_train, y_train)
    logger.info("Random Forest model trained.")

    # 5. Predict Probabilities
    probabilities = model.predict_proba(x_test)
    logger.info("Probabilities predicted.")

    filtered_output = model.filter_by_threshold(
        y_pred=probabilities,
        y_true=y_test,
        threshold=config.PROBABILITY_THRESHOLD,
        labels_encoder=le,
    )
    model.save_low_confidence_to_csv(
        x=cleaned_df["preprocess_text"],
        threshold=config.PROBABILITY_THRESHOLD,
        dataset_name=dataset_name,
        cleaned_data=cleaned_df,
    )

    logger.info("Low confidence samples saved.")

    # 6. Save Model
    models_dir = os.path.join(config.PROJECT_ROOT, "models")
    model.save_model(models_dir, dataset_name)
    model.save_tfidf_vectors(output_dir=models_dir, dataset_name=dataset_name)

    return filtered_output
