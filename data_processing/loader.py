import pandas as pd
import logging
from typing import List
import config


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
        print(f"Error loading data: {e}")
        return pd.DataFrame()
