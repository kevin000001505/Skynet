import pandas as pd
import logging
from typing import List
import config, os
import re
import chardet


def _load_and_concat_csv(
    file_paths: List[str], encoding: str = None, columns: List[str] = None
) -> pd.DataFrame:
    """
    Load multiple CSV files, apply optional encoding and column selection, and concatenate them.
    """
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path, encoding=encoding)
        if columns:
            df = df[columns]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


files_dict = config.DATA_LOCATIONS

logger = logging.getLogger(__name__)


def load_csv_data(file_name: str) -> pd.DataFrame:
    file_path = files_dict.get(file_name, [None])
    try:
        if not file_path:
            raise ValueError(f"No file paths configured for {file_name}")
        if len(file_path) == 1:
            # Single file: load directly
            return pd.read_csv(file_path[0])
        if len(file_path) == 2:
            # Two files: handle Normal dataset specially, otherwise generic concat
            if file_name == "Normal":
                return _load_and_concat_csv(
                    file_path, encoding="ISO-8859-1", columns=["text", "sentiment"]
                )
            else:
                return _load_and_concat_csv(file_path)
        # More than two paths is unexpected
        raise ValueError(f"Multiple file paths found for {file_name}: {file_path}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()


def load_big_data_csv(
    recreate: bool = False,
    output_file="data/SA.csv",
    input_dir_name="data/raw_datasets",
) -> pd.DataFrame:
    """
    Load big data CSV files.
    """
    # Check if cleaned big data file exists first before performing cleaning
    if os.path.exists(config.BIG_DATA_FILE_CLEANED) and not recreate:
        logging.info("Cleaned big data file exists. Skipping cleaning.")
        return pd.read_csv(config.BIG_DATA_FILE_CLEANED)

    else:
        try:
            os.makedirs(input_dir_name, exist_ok=True)
            if not os.path.exists(output_file):
                with open(output_file, "w") as f:
                    f.write("")

            all_files_path = get_all_files_in_directoy(input_dir_name)

            if not all_files_path:
                raise FileNotFoundError(
                    f"No files found in directory: {input_dir_name}"
                )
            logging.info(
                f"The numbers of files found in directory: {len(all_files_path)}"
            )

            big_data = merge_all_files(all_files_path)
            logging.info("Big data file loaded.")
            big_data.to_csv(config.BIG_DATA_FILE, index=False)
            return big_data

        except FileNotFoundError:
            logging.exception("Data file not found: %s", config.BIG_DATA_FILE)
            raise
        except pd.errors.EmptyDataError:
            logging.exception("CSV file is empty: %s", config.BIG_DATA_FILE)
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading big data {e}")
            raise


def get_all_files_in_directoy(file_path: str) -> List[str]:
    """
    Get all files in a directory.
    """
    results = []
    input_dir_name = os.path.join(os.path.abspath(""), file_path)
    for filename in os.listdir(input_dir_name):
        if filename.endswith(".DS_Store"):
            continue
        filepath = os.path.join(os.path.abspath(input_dir_name), filename)
        if os.path.isfile(filepath):
            results.append(filepath)
    return results


def merge_all_files(all_files_path: List[str]) -> pd.DataFrame:
    """
    Merge all files in a directory into a single DataFrame.
    """
    big_data = pd.DataFrame()

    for filepath in all_files_path:
        # Detect Encoding using chardet Library
        with open(filepath, "rb") as f:
            data = f.read()

        file_name = re.search(config.REGEX_CSV_NAME, filepath).group(1)

        encoding = chardet.detect(data)["encoding"]
        logging.info(f"{filepath} encoding: {encoding}")

        df = pd.read_csv(filepath, encoding=encoding)  # Read csv
        logging.info(f"{file_name} columns: {df.columns.tolist()}")

        if file_name in config.MAPPING["columns"]:
            df.rename(columns=config.MAPPING["columns"][file_name], inplace=True)

        logging.info(f"{file_name} columns after cleaning: {df.columns.tolist()}")
        df = df[["text", "label"]].dropna()  # Keep only relevant columns
        df = label_mapping(df, file_name, drop_neutrals=True)

        big_data = pd.concat([big_data, df])

    return big_data


def label_mapping(
    df: pd.DataFrame, file_name, drop_neutrals: bool = True
) -> pd.DataFrame:
    """
    Map labels to integers.
    """
    ret_df = df.copy()
    if drop_neutrals:
        ret_df["label"] = ret_df["label"].map(
            config.MAPPING["labels_drop_neutral"][file_name]
        )
    else:
        ret_df["label"] = ret_df["label"].map(
            config.MAPPING["labels_neutral"][file_name]
        )

    ret_df.dropna(inplace=True)
    ret_df["label"] = ret_df["label"].astype(int)
    return ret_df
