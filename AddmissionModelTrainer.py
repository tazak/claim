# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

class AdmissionModelTrainer:
    def __init__(self, data_path, model_save_path):
        # Load preprocessed data
        self.data = pd.read_csv(data_path)
        self.model_save_path = model_save_path

        # Prepare features and target for admission prediction
        self.features = self.data.drop(['ADMNS', 'Category'], axis=1)
        self.target = self.data['ADMNS']

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=0.3, random_state=42
        )

        # Initialize the XGBoost model
        self.model = XGBClassifier(eval_metric='logloss')

    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Save the trained model to a file
        joblib.dump(self.model, self.model_save_path)

        print(f"Model saved to {self.model_save_path}")

    def evaluate(self):
        # Make predictions on the test set
        predictions = self.model.predict(self.X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions)

        # Print evaluation results
        print(f"Admission Model - Accuracy: {accuracy:.4f}")
        print(f"Admission Model - Precision: {precision:.4f}")
        print(f"Admission Model - Recall: {recall:.4f}")
        print(f"Admission Model - F1 Score: {f1:.4f}")
        print("\nAdmission Model Confusion Matrix:")
        print(conf_matrix)

# Usage
if __name__ == "__main__":
    trainer = AdmissionModelTrainer('preprocessed_data.csv', 'xgb_model_admission.pkl')
    trainer.train()
    trainer.evaluate()
