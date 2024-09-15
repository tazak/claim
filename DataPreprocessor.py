import pandas as pd
from icdcodex import icd2vec, hierarchy
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import joblib
import os
import concurrent.futures


class DataPreprocessor:
    def __init__(self, df, is_training=True):
        self.data = df
        self.is_training = is_training
    
        self.G, self.icd_codes = hierarchy.icd9()
        self.embedder = icd2vec.Icd2Vec(num_embedding_dimensions=2, workers=-1)
        self.embedder.vector_size = 2

        if self.is_training:
            self.embedder.fit(self.G, self.icd_codes)
            print("hello") 
            print(f"Trained embedder vector size: {self.embedder.vector_size}")
            self.scaler = StandardScaler()
        else:
            self.embedder = joblib.load("model/Embedder.pkl")
            print(f"Trained embedder vector size: {self.embedder.vector_size}")
            self.scaler = joblib.load('model/scaler.pkl')

        print("DataPreprocessor INIT PASSED")

            
        self.diagnosis_cols = [
            'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
            'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
            'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8'
        ]
        
        if self.is_training:
            self.diagnosis_cols.append('ADMTNG_ICD9_DGNS_CD')
            
        self.sp_indicators = [
            'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD',
            'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS',
            'SP_RA_OA', 'SP_STRKETIA'
        ]
        self.payment_vars = [
            'MEDREIMB_IP', 'BENRES_IP', 'PPPYMT_IP',
            'MEDREIMB_OP', 'BENRES_OP', 'PPPYMT_OP',
            'MEDREIMB_CAR', 'BENRES_CAR', 'PPPYMT_CAR'
        ]
        self.age_group_mapping = {
            '<60': 0,
            '60-90': 1,
            '>90': 2
        }
        self.race_code_mapping = {
            1: 0,  
            2: 1,
            3: 2,
            4: 3,
            5: 4
        }
        
        if self.is_training:
            self.drop_column()
            self.create_model_directory()
            self._process_total_diagnosis_count()
            self._convert_codes_to_vectors()
            self._flatten_vectors()
            self._preprocess_other_columns()
            self._encode_columns()
            self._scale_payments()
        else:
            self.add_adm_col()
            self._process_total_diagnosis_count()
            self._convert_codes_to_vectors()
            self._flatten_vectors()
            self._preprocess_other_columns()
            self._encode_columns()
            self._scale_payments() 


    def create_model_directory(self):
        directory_path = 'model'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
      
    def drop_column(self):
        self.data= self.data.drop(columns=['SP_STATE_CODE', 'BENE_COUNTY_CD','CLM_ID'])

    def add_adm_col(self):
        self.data.rename(columns={'CLM_ID': 'isAdm'}, inplace=True)

    def _process_total_diagnosis_count(self):
        self.data['Total_Diagnosis_Count'] = self.data[self.diagnosis_cols].notna().sum(axis=1)

    def _get_immediate_parent(self, icd_code):
     if icd_code not in self.G.nodes():
        return "OTHER"
     try:
        predecessors = list(self.G.predecessors(icd_code))
        return predecessors[0] if predecessors else "OTHER"
     except nx.NetworkXError:
        return "OTHER"

    def _add_category_column(self):
     self.data['Category'] = self.data['ADMTNG_ICD9_DGNS_CD'].apply(lambda code: self._get_immediate_parent(code))

    def _code_to_vector(self, code):
        if code is None or pd.isna(code):
         vector = np.zeros(self.embedder.vector_size)
        else:
         vector = self.embedder.to_vec([code])[0] if code in self.G.nodes() else np.zeros(self.embedder.vector_size)
        return vector

    def _convert_codes_to_vectors(self):
       for col in self.diagnosis_cols: 
        self.data[col] = self.data[col].apply(lambda x: self._code_to_vector(x))
        print(self.data[col].head(5))  
       if self.is_training:
         joblib.dump(self.embedder, "model/Embedder.pkl")
        
   
    
    def _flatten_vectors(self):
        for col in self.diagnosis_cols:
            # print(f"Processing column: {col}")  # Print the column name
            # print(f"First 5 values in {col}:\n", self.data[col].head(5))  
             # Check the shape of the first element in the column
            # print(f"Shape of the first element in {col}: {np.array(self.data[col].iloc[0]).shape}")
            # if len(np.array(self.data[col].iloc[0]).shape) == 0:
            #  print(f"Column {col} contains scalar values. Replacing with [0.0, 0.0].\n")
            #  self.data[col] = self.data[col].apply(lambda x: [0.0, 0.0])
        
            vector_columns = pd.DataFrame(self.data[col].tolist(), index=self.data.index, 
                                          columns=[f'{col}_vec_{i+1}' for i in range(self.embedder.vector_size)])
            self.data = pd.concat([self.data, vector_columns], axis=1)
            self.data.drop(columns=[col], inplace=True)

    def _preprocess_other_columns(self):
        self.data['BENE_SEX_IDENT_CD'] = self.data['BENE_SEX_IDENT_CD'].map({1: 1, 2: 0})
        for col in self.sp_indicators:
            self.data[col] = self.data[col].map({1: 1, 2: 0, 0:0})
        age_bins = [0, 60, 90, 100]
        age_labels = ['<60', '60-90', '>90']
        self.data['BENE_BIRTH_DT'] = pd.to_datetime(self.data['BENE_BIRTH_DT'], format='%Y-%m-%d')
        self.data['CLM_FROM_DT'] = pd.to_datetime(self.data['CLM_FROM_DT'], format='%Y-%m-%d')
        self.data['CLM_THRU_DT'] = pd.to_datetime(self.data['CLM_THRU_DT'], format='%Y-%m-%d')
      
        self.data['AGE'] = self.data.apply(lambda e: (e['CLM_FROM_DT'] - e['BENE_BIRTH_DT']).days / 365, axis=1)
        self.data['CLM_UTLZTN_DAY_CNT'] = (self.data['CLM_THRU_DT'] - self.data['CLM_FROM_DT']).dt.days
        self.data['AGE_GROUP'] = pd.cut(self.data['AGE'], bins=age_bins, labels=age_labels, right=False)
        self.data.drop(columns=['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_DEATH_DT', 'CLM_FROM_DT', 'CLM_THRU_DT', 'PRVDR_NUM', 'BENE_HMO_CVRAGE_TOT_MONS', 'PLAN_CVRG_MOS_NUM', 'AGE'], 
                       inplace=True)
        
    def _encode_columns(self):
        self.data['AGE_GROUP'] = self.data['AGE_GROUP'].map(self.age_group_mapping)
        self.data['BENE_RACE_CD'] = self.data['BENE_RACE_CD'].map(self.race_code_mapping)
   
    def _encode_category(self):
        encoder = LabelEncoder()
        self.data['Category'] = encoder.fit_transform(self.data['Category'])

        category_mapping = pd.DataFrame({
            'Category_Name': encoder.classes_,
            'Encoded_Label': range(len(encoder.classes_))
        })
        category_mapping.to_csv('data/category_mapping.csv', index=False)

    def _scale_payments(self):
        if self.is_training:
            self.data[self.payment_vars] = self.scaler.fit_transform(self.data[self.payment_vars])
            joblib.dump(self.scaler, 'model/scaler.pkl')
        else:
            self.data[self.payment_vars] = self.scaler.transform(self.data[self.payment_vars])
 
    def save_to_csv(self, output_path):
        self.data.to_csv(output_path, index=False)

    def display_head(self):
        print(self.data.head())

if __name__ == '__main__':
    print("Started preprocessing the cms data")
    df = pd.read_csv('data/admission.csv')
    preprocessor = DataPreprocessor(df)
    preprocessor.save_to_csv('data/preprocessed_data.csv')
    preprocessor.display_head()
