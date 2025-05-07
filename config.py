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
    "Movie": [DATA_DIR + "IMDB.csv"],
    "Normal": [
        DATA_DIR + "sa_train.csv",
        DATA_DIR + "sa_test.csv",
    ],
    "Yelp": [
        DATA_DIR + "yelp_train.csv",
        DATA_DIR + "yelp_test.csv",
    ],
}

TARGET_COLUMNS = {
    "Testing": "label",
    "Movie": "label",
    "Normal": "label",
    "Yelp": "label",
    "default": "label",
}

MAPPING = {
    "columns": {
        "IMDB": {"review": "text", "sentiment": "label"},
        "sa_train": {"sentiment": "label"},
        "sa_test": {"sentiment": "label"},
        "yelp_train": {"review": "text", "sentiment": "label"},
        "yelp_test": {"review": "text", "sentiment": "label"},
    },
    "labels_drop_neutral": {
        "IMDB": {"positive": 1, "negative": 0},
        "sa_train": {"positive": 1, "negative": 0},
        "sa_test": {"positive": 1, "negative": 0},
        "yelp_train": {2: 1, 1: 0},
        "yelp_test": {2: 1, 1: 0},
    },
    "labels_neutral": {
        "IMDB": {"positive": 2, "negative": 0},
        "sa_train": {"positive": 2, "negative": 0, "neutral": 1},
        "sa_test": {"positive": 2, "negative": 0, "neutral": 1},
        "yelp_train": {2: 2, 1: 0},
        "yelp_test": {2: 2, 1: 0},
    },
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "criterion": "gini",
    "max_features": "sqrt",
    "random_state": 42,
    "max_depth": None,
    "n_jobs": -1,
    "bootstrap": True,
    "verbose": 1,
}

RANDOM_FOREST_GRID_PARAMS = {
    "rf__n_estimators": [100, 200, 500],  # Number of trees
    "rf__criterion": ["gini", "entropy"],  # Split criterion
    "rf__max_depth": [10, 20, None],  # Max depth of trees (None means unlimited)
    "rf__min_samples_split": [2, 5, 10],  # Minimum samples to split a node
    "rf__min_samples_leaf": [1, 2, 4],  # Minimum samples at a leaf node
    "rf__max_features": [
        "sqrt",
        "log2",
    ],
    "rf__bootstrap": [True],
}

N_SPLITS = 5

MAX_FEATURES = 2000
PROBABILITY_THRESHOLD = 0.8

MAX_FEATURES = 2000
TOKENIZER_MAX_LENGTH = 512

REGEX_URL = r"https?://[\w.]+"  # Slightly more robust url regex

REGEX_HTML_TAGS = r"<[^>]*>"

REGEX_HTML_TAGS2 = r"br />"

REGEX_UTF8 = r"[c|C]af\\u00e\d"

REGEX_CSV_NAME = r"\/([^\/]+)\.csv"
