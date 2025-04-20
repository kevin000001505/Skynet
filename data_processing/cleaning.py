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

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp.add_pipe("spacytextblob") Further implementation needed

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"NaN values before cleaning: {df.isna().sum()}")
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        return df

    def clean_text(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                return ""
            text = re.sub(r"https?://.+", "", text)
            doc = self.nlp(text)
            clean_tokens = [
                token.lemma_.lower()
                for token in doc
                if token.text not in string.punctuation
                and not token.is_stop
                and token.lemma_ != "-PRON-"
            ]
            if len(clean_tokens) > 0:
                return " ".join(clean_tokens)
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text


class MovieDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class NormalTextCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        df["preprocess_text"] = df["text"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class TwitterDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class YelpDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        df["sentiment"] = df["sentiment"].map({2: "positive", 1: "negative"})
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class TestingDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df
