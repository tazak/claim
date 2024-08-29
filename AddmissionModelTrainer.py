import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

class ModelTrainer:
    def __init__(self, data_path, adm_model_save_path, cat_model_save_path):
        self.data = pd.read_csv(data_path )
        self.adm_model_save_path = adm_model_save_path
        self.cat_model_save_path = cat_model_save_path

        # features and target for admission prediction
        self.features_adm = self.data.drop(['ADMNS', 'Category'], axis=1)
        self.target_adm = self.data['ADMNS']

        # features and target for category prediction
        self.features_cat = self.data.drop(['ADMNS', 'Category',
            'ICD9_DGNS_CD_1_vec_1', 'ICD9_DGNS_CD_1_vec_2', 'ICD9_DGNS_CD_2_vec_1',
            'ICD9_DGNS_CD_2_vec_2', 'ICD9_DGNS_CD_3_vec_1', 'ICD9_DGNS_CD_3_vec_2',
            'ICD9_DGNS_CD_4_vec_1', 'ICD9_DGNS_CD_4_vec_2', 'ICD9_DGNS_CD_5_vec_1',
            'ICD9_DGNS_CD_5_vec_2', 'ICD9_DGNS_CD_6_vec_1', 'ICD9_DGNS_CD_6_vec_2',
            'ICD9_DGNS_CD_7_vec_1', 'ICD9_DGNS_CD_7_vec_2', 'ICD9_DGNS_CD_8_vec_1',
            'ICD9_DGNS_CD_8_vec_2'], axis=1)
        self.target_cat = self.data['Category']

        # training and testing sets
        self.X_train_adm, self.X_test_adm, self.y_train_adm, self.y_test_adm = train_test_split(
            self.features_adm, self.target_adm, test_size=0.3, random_state=42, stratify=self.target_adm
        )
        self.X_train_cat, self.X_test_cat, self.y_train_cat, self.y_test_cat = train_test_split(
            self.features_cat, self.target_cat, test_size=0.3, random_state=42, stratify=self.target_cat
        )

        # Initializing XGBoost models
        self.model_adm = XGBClassifier(eval_metric='logloss')
        self.model_cat = XGBClassifier(eval_metric='mlogloss')

    def train_admission_model(self):
        self.model_adm.fit(self.X_train_adm, self.y_train_adm)
        joblib.dump(self.model_adm, self.adm_model_save_path)
        print(f"Admission Model saved to {self.adm_model_save_path}")

    def evaluate_admission_model(self):
        predictions_adm = self.model_adm.predict(self.X_test_adm)
        # metrics
        accuracy_adm = accuracy_score(self.y_test_adm, predictions_adm)
        precision_adm = precision_score(self.y_test_adm, predictions_adm)
        recall_adm = recall_score(self.y_test_adm, predictions_adm)
        f1_adm = f1_score(self.y_test_adm, predictions_adm)
        conf_matrix_adm = confusion_matrix(self.y_test_adm, predictions_adm)

        # results
        print(f"Admission Model - Accuracy: {accuracy_adm:.4f}")
        print(f"Admission Model - Precision: {precision_adm:.4f}")
        print(f"Admission Model - Recall: {recall_adm:.4f}")
        print(f"Admission Model - F1 Score: {f1_adm:.4f}")
        print("\nAdmission Model Confusion Matrix:")
        print(conf_matrix_adm)

    def predict_single_record(self, record_df):
     if isinstance(record_df, np.ndarray):
        record_df = pd.DataFrame(record_df, columns=self.features_adm.columns)
     elif not isinstance(record_df, pd.DataFrame):
        raise ValueError("Input record must be a DataFrame or ndarray.")
     features = self.features_adm.columns
     if not set(features).issubset(record_df.columns):
        raise ValueError("Input record does not have the correct feature columns.")
     record_df = record_df[features]
     prediction = self.model_adm.predict(record_df)
     return prediction[0]
 
 
    
    def train_category_model(self):
        self.model_cat.fit(self.X_train_cat, self.y_train_cat)

        joblib.dump(self.model_cat, self.cat_model_save_path)
        print(f"Category Model saved to {self.cat_model_save_path}")

    def evaluate_category_model(self):
        predictions_cat = self.model_cat.predict(self.X_test_cat)

        # metrics
        accuracy_cat = accuracy_score(self.y_test_cat, predictions_cat)
        precision_cat = precision_score(self.y_test_cat, predictions_cat, average='weighted')
        recall_cat = recall_score(self.y_test_cat, predictions_cat, average='weighted')
        f1_cat = f1_score(self.y_test_cat, predictions_cat, average='weighted')
        conf_matrix_cat = confusion_matrix(self.y_test_cat, predictions_cat)

        # results
        print(f"Category Model - Accuracy: {accuracy_cat:.4f}")
        print(f"Category Model - Precision: {precision_cat:.4f}")
        print(f"Category Model - Recall: {recall_cat:.4f}")
        print(f"Category Model - F1 Score: {f1_cat:.4f}")
        print("\nCategory Model Confusion Matrix:")
        print(conf_matrix_cat)

# Usage
if __name__ == "__main__":
    trainer = ModelTrainer('preprocessed_data.csv', 'xgb_model_admission.pkl', 'xgb_model_category.pkl')
    trainer.train_admission_model()
    trainer.evaluate_admission_model()
    trainer.train_category_model()
    trainer.evaluate_category_model()
