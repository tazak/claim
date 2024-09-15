import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import mlflow

class ModelTrainer:
    def __init__(self, data_path, adm_model_save_path, cat_model_save_path):
        # MLflow tracking URI for logging
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("xgboost")

        self.data = pd.read_csv(data_path)
        self.data = pd.read_csv(data_path)
        print("Data columns:", self.data.columns)

        self.adm_model_save_path = adm_model_save_path
        self.cat_model_save_path = cat_model_save_path

        # Features and target for admission prediction
        self.features_adm = self.data.drop(['isAdm','ADMTNG_ICD9_DGNS_CD_vec_1','ADMTNG_ICD9_DGNS_CD_vec_2'], axis=1)
        self.target_adm = self.data['isAdm']

        # Features and target for ADM_ICD prediction
        self.target_cat = self.data[['ADMTNG_ICD9_DGNS_CD_vec_1','ADMTNG_ICD9_DGNS_CD_vec_2']]
        self.features_cat = self.data.drop(['ADMTNG_ICD9_DGNS_CD_vec_1','ADMTNG_ICD9_DGNS_CD_vec_2'], axis=1)

        # Stratified splitting 
        self.X_train_adm, self.X_test_adm, self.y_train_adm, self.y_test_adm = train_test_split(
            self.features_adm, self.target_adm, test_size=0.3, random_state=42, stratify=self.target_adm
        )

        self.X_train_cat, self.X_test_cat, self.y_train_cat, self.y_test_cat = train_test_split(
            self.features_cat, self.target_cat, test_size=0.3, random_state=42
        )

        # XGBoost models
        self.model_adm = XGBClassifier(eval_metric='logloss')
        self.model_cat = MultiOutputRegressor(XGBRegressor(eval_metric='rmse'))

    def save_adm_test_data(self, adm_test_path='adm_test_data.csv'):
        adm_test_data = pd.DataFrame(self.X_test_adm)
        adm_test_data.to_csv(adm_test_path, index=False)
        print(f"Admission Test data saved to {adm_test_path}")

    def train_admission_model(self):
        self.model_adm.fit(self.X_train_adm, self.y_train_adm)
        joblib.dump(self.model_adm, self.adm_model_save_path)
        print(f"Admission Model saved to {self.adm_model_save_path}")

    def evaluate_admission_model(self):
        predictions_adm = self.model_adm.predict(self.X_test_adm)
        accuracy_adm = accuracy_score(self.y_test_adm, predictions_adm)
        precision_adm = precision_score(self.y_test_adm, predictions_adm)
        recall_adm = recall_score(self.y_test_adm, predictions_adm)
        f1_adm = f1_score(self.y_test_adm, predictions_adm)
        conf_matrix_adm = confusion_matrix(self.y_test_adm, predictions_adm)

        # Log metrics
        mlflow.log_metric("accuracy_adm", accuracy_adm)
        mlflow.log_metric("precision_adm", precision_adm)
        mlflow.log_metric("recall_adm", recall_adm)
        mlflow.log_metric("f1_adm", f1_adm)

        # evaluation results
        print(f"Admission Model - Accuracy: {accuracy_adm:.4f}")
        print(f"Admission Model - Precision: {precision_adm:.4f}")
        print(f"Admission Model - Recall: {recall_adm:.4f}")
        print(f"Admission Model - F1 Score: {f1_adm:.4f}")
        print("\nAdmission Model Confusion Matrix:")
        print(conf_matrix_adm)

    def train_ADM_ICD_model(self):
        self.model_cat.fit(self.X_train_cat, self.y_train_cat)
        joblib.dump(self.model_cat, self.cat_model_save_path)
        print(f"ADM_ICD Model saved to {self.cat_model_save_path}")

    def evaluate_ADM_ICD_model(self):
        predictions_cat = self.model_cat.predict(self.X_test_cat)

        # Mean Squared Error for both dimensions of the vectors
        mse_cat = mean_squared_error(self.y_test_cat, predictions_cat, multioutput='raw_values')
        mae_cat = mean_absolute_error(self.y_test_cat, predictions_cat, multioutput='raw_values')
        r2_cat = r2_score(self.y_test_cat, predictions_cat, multioutput='raw_values')

        #Log metrics
        mlflow.log_metric("mse_cat_dim1", mse_cat[0])
        mlflow.log_metric("mse_cat_dim2", mse_cat[1])
        mlflow.log_metric("mae_cat_dim1", mae_cat[0])
        mlflow.log_metric("mae_cat_dim2", mae_cat[1])
        mlflow.log_metric("r2_cat_dim1", r2_cat[0])
        mlflow.log_metric("r2_cat_dim2", r2_cat[1])

        #evaluation results
        print(f"ADM_ICD Model - MSE (Dimension 1): {mse_cat[0]:.4f}")
        print(f"ADM_ICD  Model - MSE (Dimension 2): {mse_cat[1]:.4f}")
        print(f"ADM_ICD  Model - MAE (Dimension 1): {mae_cat[0]:.4f}")
        print(f"ADM_ICD  Model - MAE (Dimension 2): {mae_cat[1]:.4f}")
        print(f"ADM_ICD  Model - R2 (Dimension 1): {r2_cat[0]:.4f}")
        print(f"ADM_ICD  Model - R2 (Dimension 2): {r2_cat[1]:.4f}")

# Usage
if __name__ == "__main__":
    trainer = ModelTrainer('data/preprocessed_data.csv', 'model/xgb_model_admission.pkl', 'model/xgb_model_ADM_ICD.pkl')
    trainer.train_admission_model()
    print("Admission model training done")
    trainer.evaluate_admission_model()
    print("Admission model evaluation done")
    trainer.save_adm_test_data()
    trainer.train_ADM_ICD_model()
    print("ADM_ICD model training done")
    trainer.evaluate_ADM_ICD_model()
    print("ADM_ICD model evaluation done")
