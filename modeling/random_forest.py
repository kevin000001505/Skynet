import numpy as np
import logging
import os
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

logger = logging.getLogger(__name__)
default_params = config.RANDOM_FOREST_PARAMS


class RandomForestModel:
    def __init__(self, params=default_params):
        self.pipeline = Pipeline(
            [("tfidf", TfidfVectorizer()), ("rf", RandomForestClassifier(**params))]
        )

    def train(self, x_train, y_train):
        self.pipeline.fit(x_train, y_train)
        self.model_class = self.pipeline.named_steps["rf"].classes_

    def predict(self, x_test):
        return self.pipeline.predict(x_test)

    def predict_proba(self, x_test):
        return self.pipeline.predict_proba(x_test)

    def filter_by_threshold(self, y_pred, y_true, threshold=0.5, labels_encoder=None):
        confidence_mask = np.any(y_pred > threshold, axis=1)
        true_labels_filtered = y_true[confidence_mask]

        filtered_probabilities = y_pred[confidence_mask]
        predicted_labels_filtered = self.model_class[
            np.argmax(filtered_probabilities, axis=1)
        ]

        return {
            "y_true": labels_encoder.inverse_transform(true_labels_filtered),
            "y_pred": labels_encoder.inverse_transform(predicted_labels_filtered),
            "threshold": threshold,
        }

    def save_low_confidence_to_csv(
        self,
        x,
        threshold=0.5,
        csv_path="low_confidence_samples/{dataset_name}.csv",
        dataset_name="dataset",
        raw_data=None,
    ):
        csv_path = csv_path.format(dataset_name=dataset_name)
        y_pred = self.pipeline.predict_proba(x)
        high_confidence_mask = np.max(y_pred, axis=1) > threshold
        high_confidence_indexs = np.nonzero(high_confidence_mask)[0].tolist()
        low_confidence_data = raw_data.drop(high_confidence_indexs).reset_index(
            drop=True
        )

        try:
            low_confidence_data.to_csv(csv_path, index=False)
            logging.info(f"{dataset_name}: Low confidence samples saved to {csv_path}")
        except Exception as e:
            logging.error(f"Error saving low confidence samples: {e}")

    def save_model(self, output_dir, dataset_name):
        """
        Save the trained RandomForest pipeline to disk.
        :param output_dir: Directory where model file will be saved
        :param dataset_name: Used to name the model file
        """
        os.makedirs(output_dir, exist_ok=True)
        model_filename = f"{dataset_name}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(self.pipeline, model_path)
        logger.info(f"Random Forest pipeline saved to {model_path}")
