import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from collections import defaultdict


class FraudDetectionModel:
    def __init__(self):
        self.label_encoders = {}
        self.label_maps = defaultdict(dict)  # Store category mappings for each feature
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=200,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric=["error", "logloss"],
            enable_categorical=True,
        )

    def safe_label_encode(self, series, feature_name, is_training=True):
        """Safely encode labels, handling unseen categories"""
        if is_training:
            # Training phase: create a new encoder and mapping
            unique_values = series.unique()
            self.label_maps[feature_name] = {
                val: i for i, val in enumerate(unique_values)
            }
            return series.map(self.label_maps[feature_name])
        else:
            # Testing phase: use existing mapping and map unseen categories to a special value
            mapping = self.label_maps[feature_name]
            max_val = max(mapping.values())
            return series.map(lambda x: mapping.get(x, max_val + 1))

    def convert_time_to_minutes(self, time_str):
        """Convert a time string to minutes past midnight"""
        try:
            hours, minutes = map(int, time_str.split(":"))
            return hours * 60 + minutes
        except:
            print(f"Unable to parse time format: {time_str}")
            return 0  # Return default value instead of None

    def preprocess_data(self, df, is_training=True):
        # Create a copy of the data
        data = df.copy()

        # Handle categorical features
        categorical_features = ["att3", "att6", "att7", "att8", "att9"]

        for feature in categorical_features:
            data[feature] = self.safe_label_encode(data[feature], feature, is_training)

        # Process time features
        data["minutes_from_midnight"] = data["att1"].apply(self.convert_time_to_minutes)
        data["hour"] = data["minutes_from_midnight"] // 60

        # Create time period feature
        data["time_period"] = pd.cut(
            data["hour"],
            bins=[-1, 5, 11, 16, 21, 24],
            labels=["late_night", "morning", "afternoon", "evening", "night"],
        )
        data["time_period"] = self.safe_label_encode(
            data["time_period"], "time_period", is_training
        )

        # Create cyclic time features
        minutes_in_day = 24 * 60
        data["time_sin"] = np.sin(
            2 * np.pi * data["minutes_from_midnight"] / minutes_in_day
        )
        data["time_cos"] = np.cos(
            2 * np.pi * data["minutes_from_midnight"] / minutes_in_day
        )

        # Calculate geographic distance
        data["distance"] = np.sqrt(
            (data["att12"] - data["att15"]) ** 2 + (data["att13"] - data["att16"]) ** 2
        )

        # Feature engineering
        data["amount_per_distance"] = data["att4"] / (
            data["distance"] + 1
        )  # Avoid division by zero
        data["amount_time_factor"] = data["att4"] * np.abs(
            data["time_sin"]
        )  # Relationship between transaction amount and time

        # Select features to use
        features = [
            "att4",  # Transaction amount
            "att5",  # Cardholder age
            "hour",  # Hour
            "time_period",  # Time period
            "time_sin",  # Cyclic time feature (sine)
            "time_cos",  # Cyclic time feature (cosine)
            "distance",  # Geographic distance
            "att10",  # City population
            "att3",  # Transaction category
            "att6",  # Gender
            "att7",  # Occupation
            "att8",  # City
            "att9",  # State
            "amount_per_distance",  # Transaction amount per unit distance
            "amount_time_factor",  # Transaction amount-time factor
        ]

        X = data[features]

        # Standardize numerical features
        numerical_features = [
            "att4",
            "att5",
            "distance",
            "att10",
            "time_sin",
            "time_cos",
            "amount_per_distance",
            "amount_time_factor",
        ]
        if is_training:
            self.scaler.fit(X[numerical_features])
        X[numerical_features] = self.scaler.transform(X[numerical_features])

        return X

    def train(self, train_data):
        print("Training model...")

        # Preprocess training data
        X = self.preprocess_data(train_data, is_training=True)
        y = train_data["fraud"]

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Compute class weights
        total = len(y_train)
        fraud_ratio = sum(y_train) / total
        scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
        self.model.set_params(scale_pos_weight=scale_pos_weight)

        # Train model
        eval_set = [(X_val, y_val)]
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(
            f"\nCross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Evaluate model on validation set
        y_pred = self.model.predict(X_val)
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

        # Output feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": self.model.feature_importances_}
        )
        print("\nFeature Importance:")
        print(feature_importance.sort_values("importance", ascending=False))

    def predict(self, test_data):
        # Preprocess test data
        X_test = self.preprocess_data(test_data, is_training=False)

        # Make predictions
        predictions = self.model.predict(X_test)

        # Create submission file
        submission = pd.DataFrame({"Id": test_data["Id"], "fraud": predictions})

        return submission


# Example Usage
if __name__ == "__main__":
    # Load data
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")

    # Create and train model
    model = FraudDetectionModel()
    model.train(train_data)

    # Make predictions and save results
    predictions = model.predict(test_data)
    predictions.to_csv("submission.csv", index=False)
    print("\nPredictions saved to 'submission.csv'")
