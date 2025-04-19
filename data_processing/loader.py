import pandas as pd
import logging
from typing import List
import config

files_dict = config.DATA_LOCATIONS

logger = logging.getLogger(__name__)


def load_csv_data(file_name: str) -> pd.DataFrame:
    file_path = files_dict.get(file_name, [None])
    try:
        if len(file_path) == 1:
            data = pd.read_csv(file_path[0])
            return data
        elif len(file_path) == 2:
            data1 = pd.read_csv(file_path[0])
            data2 = pd.read_csv(file_path[1])
            combined_data = pd.concat([data1, data2], ignore_index=True)
            return combined_data
        else:
            raise ValueError(f"Multiple file paths found for {file_name}: {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
