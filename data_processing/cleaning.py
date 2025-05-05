import pandas as pd
from tqdm.auto import tqdm
import logging, config

import spacy
import string
import re

logger = logging.getLogger(__name__)
tqdm.pandas()


CLEANING_TEXT_MESSAGE = "Cleaning text data..."


class BaseDataCleaner:

    def __init__(self, batch_size=300):
        self.nlp = spacy.load(
            "en_core_web_sm", disable=["ner", "parser", "senter", "textcat"]
        )
        self.batch_size = batch_size

    def _process_doc(self, doc):
        clean_tokens = [
            f"{token.lemma_.lower()}"
            for token in doc
            if token.text not in string.punctuation and not token.is_stop
        ]
        if len(clean_tokens) > 0:
            return " ".join(clean_tokens)
        return doc.text

    def removing_special_characters(
        self, df: pd.DataFrame, text_columns: str
    ) -> pd.DataFrame:
        """
        Remove special characters from the specified text columns in the DataFrame.
        """
        df[text_columns] = df[text_columns].astype(str)
        df[text_columns] = df[text_columns].apply(
            lambda x: re.sub(config.REGEX_URL, "", x)
        )
        df[text_columns] = df[text_columns].apply(
            lambda x: re.sub(config.REGEX_HTML_TAGS, "", x)
        )
        df[text_columns] = df[text_columns].apply(
            lambda x: re.sub(config.REGEX_HTML_TAGS2, "", x)
        )
        df[text_columns] = df[text_columns].apply(
            lambda x: re.sub(config.REGEX_UTF8, "", x)
        )
        return df

    def big_data_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = self.clean(df)
        print(CLEANING_TEXT_MESSAGE)
        df = self.removing_special_characters(df, "text")
        text_list = df["text"].tolist()
        for doc in tqdm(
            self.nlp.pipe(
                text_list,
                batch_size=self.batch_size,
                n_process=8,
            ),
            total=len(text_list),
        ):
            processed_texts.append(self._process_doc(doc))

        df["preprocess_text"] = processed_texts
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        df.to_csv(config.BIG_DATA_FILE_CLEANED, index=False)
        print("Cleaner: Cleaned big data file writen to data/SA_cleaned.csv.")
        return df


class MovieDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        texts = (
            df["review"].astype(str).apply(lambda x: re.sub(config.REGEX_URL, "", x))
        )
        text_list = texts.tolist()
        for doc in tqdm(
            self.nlp.pipe(
                text_list,
                batch_size=self.batch_size,
                n_process=8,
            ),
            total=len(text_list),
        ):
            processed_texts.append(self._process_doc(doc))

        df["preprocess_text"] = processed_texts
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class NormalTextCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        texts = df["text"].astype(str).apply(lambda x: re.sub(config.REGEX_URL, "", x))
        text_list = texts.tolist()
        for doc in tqdm(
            self.nlp.pipe(
                text_list,
                batch_size=self.batch_size,
                n_process=8,
            ),
            total=len(text_list),
        ):
            processed_texts.append(self._process_doc(doc))

        df["preprocess_text"] = processed_texts
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class YelpDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = super().clean(df)
        df["sentiment"] = df["sentiment"].map({1: "positive", 0: "negative"})
        print(CLEANING_TEXT_MESSAGE)
        texts = (
            df["review"].astype(str).apply(lambda x: re.sub(config.REGEX_URL, "", x))
        )
        text_list = texts.tolist()
        for doc in tqdm(
            self.nlp.pipe(
                text_list,
                batch_size=self.batch_size,
                n_process=8,
            ),
            total=len(text_list),
        ):
            processed_texts.append(self._process_doc(doc))

        df["preprocess_text"] = processed_texts
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class TestingDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        texts = df["text"].astype(str).apply(lambda x: re.sub(config.REGEX_URL, "", x))
        text_list = texts.tolist()
        for doc in tqdm(
            self.nlp.pipe(
                text_list,
                batch_size=self.batch_size,
                n_process=8,
            ),
            total=len(text_list),
        ):
            processed_texts.append(self._process_doc(doc))

        df["preprocess_text"] = processed_texts
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df
