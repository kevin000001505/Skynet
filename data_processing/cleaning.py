import pandas as pd
from tqdm import tqdm

# from spacytextblob.spacytextblob import SpacyTextBlob
import spacy
import string
import re

tqdm.pandas()


CLEANING_TEXT_MESSAGE = "Cleaning text data..."


class BaseDataCleaner:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp.add_pipe("spacytextblob")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(inplace=True)  # Drop rows with missing values
        df.drop_duplicates(inplace=True)  # Drop duplicate rows
        return df

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"https?://.+", "", text)
        doc = self.nlp(text)
        clean_tokens = [
            token.lemma_.lower()
            for token in doc
            if token.text not in string.punctuation  # remove punctuation
            and not token.is_stop  # remove stopwords
            and token.lemma_ != "-PRON-"  # exclude pronouns
        ]
        return " ".join(clean_tokens)


class MovieDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        # df["preprocess_text"] = df["review"].apply(self.clean_text)
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class NormalTextCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        # df["preprocess_text"] = df["text"].apply(self.clean_text)
        df["preprocess_text"] = df["text"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class TwitterDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        # df["preprocess_text"] = df["review"].apply(self.clean_text)
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class YelpDataCleaner(BaseDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        # df["preprocess_text"] = df["review"].apply(self.clean_text)
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df


class TestingDataCleaner(BaseDataCleaner):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().clean(df)
        print(CLEANING_TEXT_MESSAGE)
        # df["preprocess_text"] = df["review"].apply(self.clean_text)
        df["preprocess_text"] = df["review"].progress_apply(self.clean_text)
        df = df[df["preprocess_text"].str.split().str.len() > 0]
        return df
