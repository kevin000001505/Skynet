import pandas as pd
from tqdm.auto import tqdm
import logging

# from spacytextblob.spacytextblob import SpacyTextBlob
import spacy
import string
import re

logger = logging.getLogger(__name__)
tqdm.pandas()


CLEANING_TEXT_MESSAGE = "Cleaning text data..."


class BaseDataCleaner:

    def __init__(self, batch_size=500):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.batch_size = batch_size
        # self.nlp.add_pipe("spacytextblob") Further implementation needed

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"NaN values before cleaning: {df.isna().sum()}")
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        return df

    def _process_doc(self, doc):
        clean_tokens = [
            token.lemma_.lower()
            for token in doc
            if token.text not in string.punctuation
            and not token.is_stop
            and token.lemma_ != "-PRON-"
        ]
        if len(clean_tokens) > 0:
            return " ".join(clean_tokens)
        return doc.text

    def big_data_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = self.clean(df)
        print(CLEANING_TEXT_MESSAGE)
        texts = df["text"].astype(str).apply(lambda x: re.sub(r"https?://.+", "", x))
        text_list = texts.tolist()
        for doc in tqdm(
            self.nlp.pipe(
                text_list,
                batch_size=300,
                n_process=8,
            ),
            total=len(text_list),
        ):
            processed_texts.append(self._process_doc(doc))

        df["preprocess_text"] = processed_texts
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class MovieDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        texts = df["review"].astype(str).apply(lambda x: re.sub(r"https?://.+", "", x))
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
        texts = df["text"].astype(str).apply(lambda x: re.sub(r"https?://.+", "", x))
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


class TwitterDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_texts = []
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        texts = df["review"].astype(str).apply(lambda x: re.sub(r"https?://.+", "", x))
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
        df["sentiment"] = df["sentiment"].map({2: "positive", 1: "negative"})
        print(CLEANING_TEXT_MESSAGE)
        texts = df["review"].astype(str).apply(lambda x: re.sub(r"https?://.+", "", x))
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
        texts = df["review"].astype(str).apply(lambda x: re.sub(r"https?://.+", "", x))
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
