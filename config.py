import os

from data_processing.cleaning import (
    MovieDataCleaner,
    NormalTextCleaner,
    YelpDataCleaner,
    BaseDataCleaner,
    TestingDataCleaner,  # Testing data cleaner
)  # Import specific cleaners

DATA_DIR = "data/raw_datasets/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BIG_DATA_FILE = "data/SA.csv"
BIG_DATA_FILE_CLEANED = "data/SA_cleaned.csv"
DEBUG = False

CLEANING_STRATEGIES = {
    "Testing": TestingDataCleaner,  # Testing data cleaner
    "Movie": MovieDataCleaner,
    "Normal": NormalTextCleaner,
    "Yelp": YelpDataCleaner,
    "default": BaseDataCleaner,
}


DATA_LOCATIONS = {
    # "Testing": [DATA_DIR + "testing.csv"],
    "Twitter": [DATA_DIR + "Twitter Training Data.csv"],
    "Movie": [DATA_DIR + "IMDB Dataset.csv"],
    "Normal": [
        DATA_DIR + "train.csv",
        DATA_DIR + "test.csv",
    ],
    "Yelp": [
        DATA_DIR + "yelp_review_polarity_csv/train.csv",
        DATA_DIR + "yelp_review_polarity_csv/test.csv",
    ],
}

TARGET_COLUMNS = {
    "Testing": "label",
    "Twitter": "label",
    "Movie": "label",
    "Normal": "label",
    "Yelp": "label",
    "default": "label",
}

RANDOM_FOREST_PARAMS = {
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 1,
}
MAX_FEATURES = 2000
PROBABILITY_THRESHOLD = 0.8

MAX_FEATURES = 2000
TOKENIZER_MAX_LENGTH = 512

REGEX_URL = r"https?://[\w.]+"  # Slightly more robust url regex
REGEX_HTML_TAGS = {
    "Yelp": [r"<[^>]*>", r"br />"],
}
REGEX_UTF8 = {
    "Yelp": r"[c|C]af\\u00e\d",
}
