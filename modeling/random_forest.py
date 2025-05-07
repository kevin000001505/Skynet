import numpy as np
import logging
import os
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)
default_params = config.RANDOM_FOREST_PARAMS
max_features = config.MAX_FEATURES


class RandomForestModel:
    def __init__(self, params=default_params):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features)),
                ("rf", RandomForestClassifier(**params)),
            ]
        )
        self.params = params

    def _get_cv_search_object(self):
        return RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=config.RANDOM_FOREST_GRID_PARAMS,
            n_iter=10,  # Number of parameter settings that are sampled
            cv=3,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
            verbose=3,
            refit=True,
            return_train_score=True,
        )

    def train(self, x_train, y_train):
        if self.params is None:
            logger.info("Starting RandomizedSearchCV for hyperparameter tuning")
            random_search = self._get_cv_search_object()
            random_search.fit(x_train, y_train)
            self.pipeline = random_search.best_estimator_
        else:
            self.pipeline.fit(x_train, y_train)
        self.model_class = self.pipeline.named_steps["rf"].classes_

    def predict(self, x_test):
        return self.pipeline.predict(x_test)

    def predict_proba(self, x_test):
        return self.pipeline.predict_proba(x_test)

    def filter_by_threshold(self, y_pred, y_true, threshold=0.5, labels_encoder=None):
        confidence_mask = np.max(y_pred, axis=1) > threshold
        true_labels_filtered = y_true[confidence_mask]

        filtered_probabilities = y_pred[confidence_mask]
        predicted_labels_filtered = self.model_class[
            np.argmax(filtered_probabilities, axis=1)
        ]
        coverage_percentage = 100 * np.sum(confidence_mask) / len(y_pred)

        return {
            "y_true": labels_encoder.inverse_transform(true_labels_filtered),
            "y_pred": labels_encoder.inverse_transform(predicted_labels_filtered),
            "threshold": threshold,
            "coverage_percentage": coverage_percentage,
        }

    def save_low_confidence_to_csv(
        self,
        x,
        cleaned_data=None,
        csv_path=config.BIG_DATA_FILE_CLEANED,
    ):
        y_pred = self.pipeline.predict_proba(x)
        cleaned_data["confidence"] = np.max(y_pred, axis=1).tolist()
        cleaned_data.to_csv(csv_path, index=False)
        logger.info(f"Low confidence samples saved to {csv_path}")

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
