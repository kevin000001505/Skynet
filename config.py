from data_processing.cleaning import (
    MovieDataCleaner,
    NormalTextCleaner,
    TwitterDataCleaner,
    YelpDataCleaner,
    BaseDataCleaner,
    TestingDataCleaner,  # Testing data cleaner
)  # Import specific cleaners

DATA_DIR = "data/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/"
PROJECT_ROOT = "."


CLEANING_STRATEGIES = {
    "Testing": TestingDataCleaner,  # Testing data cleaner
    "Twitter": TwitterDataCleaner,
    "Movie": MovieDataCleaner,
    "Normal": NormalTextCleaner,
    "Yelp": YelpDataCleaner,
    "default": BaseDataCleaner,
}


DATA_LOCATIONS = {
    # "Testing": [DATA_DIR + "testing.csv"],
    # "Twitter": [DATA_DIR + "Twitter Training Data.csv"],
    # "Movie": [DATA_DIR + "IMDB Dataset.csv"],
    # "Normal": [
    #     DATA_DIR + "train.csv",
    #     DATA_DIR + "test.csv",
    # ],
    "Yelp": [
        DATA_DIR + "yelp_review_polarity_csv/train.csv",
        DATA_DIR + "yelp_review_polarity_csv/test.csv",
    ],
}

TARGET_COLUMNS = {
    "Testing": "sentiment",
    "Twitter": "sentiment",
    "Movie": "sentiment",
    "Normal": "sentiment",
    "Yelp": "sentiment",
    "default": "sentiment",
}

RANDOM_FOREST_PARAMS = {
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}
PROBABILITY_THRESHOLD = 0.7
