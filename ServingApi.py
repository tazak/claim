from typing import Optional
from pydantic import BaseModel, Field
import pandas as pd
from fastapi import FastAPI, HTTPException
import joblib
from DataPreprocessor import DataPreprocessor

app = FastAPI(
    title="Admission Prediction App",
    description="This API predicts whether a patient will be admitted based on provided medical records.",
    version="1.0.0"
)

adm_model_path = "model/xgb_model_admission.pkl"  
adm_ICD_model_path = "model/xgb_model_ADM_ICD.pkl"  
embedder_path ="model/Embedder.pkl"

try:
    global admission_model, ADM_ICD_model
    admission_model = joblib.load(adm_model_path)
    ADM_ICD_model = joblib.load(adm_ICD_model_path)
    embedder = joblib.load(embedder_path)
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")


# Pydantic model for request validation with descriptions
class Record(BaseModel):
    DESYNPUF_ID: str = Field(..., description="Beneficiary Code, a unique identifier for each beneficiary.")
    BENE_BIRTH_DT: str = Field(..., description="Date of birth in YYYY-MM-DD format.", pattern=r"^\d{4}-\d{2}-\d{2}$")
    BENE_DEATH_DT: Optional[str] = Field(None, description="Date of death in YYYY-MM-DD format. Can be null if the beneficiary is alive.", pattern=r"^\d{4}-\d{2}-\d{2}$|^NULL$")
    BENE_SEX_IDENT_CD: int = Field(..., description="Sex of the beneficiary. 1: Male, 2: Female.", ge=1, le=2)
    BENE_RACE_CD: int = Field(..., description="Beneficiary Race Code. An integer code representing the race.", ge=1, le=5)
    BENE_ESRD_IND: int = Field(..., description="End stage renal disease Indicator. '1' for Yes, '0' for No.",ge=0, le=2)
    BENE_HI_CVRAGE_TOT_MONS: int = Field(..., description="Total number of months of part A coverage for the beneficiary.", ge=0)
    BENE_SMI_CVRAGE_TOT_MONS: int = Field(..., description="Total number of months of part B coverage for the beneficiary.", ge=0)
    BENE_HMO_CVRAGE_TOT_MONS: int = Field(..., description="Total number of months of HMO coverage for the beneficiary.", ge=0)
    PLAN_CVRG_MOS_NUM: int = Field(..., description="Total number of months of part D plan coverage for the beneficiary.", ge=0)
    SP_ALZHDMTA: int = Field(..., description="Indicator for Alzheimer's or related disorders.", ge=0, le=2)
    SP_CHF: int = Field(..., description="Indicator for Chronic Heart Failure.", ge=0, le=2)
    SP_CHRNKIDN: int = Field(..., description="Indicator for Chronic Kidney Disease.", ge=0, le=2)
    SP_CNCR: int = Field(..., description="Indicator for Cancer.", ge=0, le=2)
    SP_COPD: int = Field(..., description="Indicator for Chronic Obstructive Pulmonary Disease.", ge=0, le=2)
    SP_DEPRESSN: int = Field(..., description="Indicator for Depression.", ge=0, le=2)
    SP_DIABETES: int = Field(..., description="Indicator for Diabetes.", ge=0, le=2)
    SP_ISCHMCHT: int = Field(..., description="Indicator for Ischemic Heart Disease.", ge=0, le=2)
    SP_OSTEOPRS: int = Field(..., description="Indicator for Osteoporosis.", ge=0, le=2)
    SP_RA_OA: int = Field(..., description="Indicator for Rheumatoid Arthritis and Osteoarthritis.", ge=0, le=2)
    SP_STRKETIA: int = Field(..., description="Indicator for Stroke/Transient Ischemic Attack.", ge=0, le=2)
    MEDREIMB_IP: int = Field(..., description="Inpatient annual Medicare reimbursement amount.", ge=0)
    BENRES_IP: int = Field(..., description="Inpatient annual beneficiary responsibility amount.", ge=0)
    PPPYMT_IP: int = Field(..., description="Inpatient annual primary payer reimbursement amount.", ge=0)
    MEDREIMB_OP: int = Field(..., description="Outpatient Institutional annual Medicare reimbursement amount.", ge=0)
    BENRES_OP: int = Field(..., description="Outpatient Institutional annual beneficiary responsibility amount.", ge=0)
    PPPYMT_OP: int = Field(..., description="Outpatient Institutional annual primary payer reimbursement amount.", ge=0)
    MEDREIMB_CAR: int = Field(..., description="Carrier annual Medicare reimbursement amount.", ge=0)
    BENRES_CAR: int = Field(..., description="Carrier annual beneficiary responsibility amount.", ge=0)
    PPPYMT_CAR: int = Field(..., description="Carrier annual primary payer reimbursement amount.", ge=0)
    CLM_ID: int = Field(..., description="Claim ID, a unique identifier for each claim.")
    ADMTNG_ICD9_DGNS_CD: Optional[str] = Field(None, description="Claim Admitting Diagnosis Code")
    CLM_FROM_DT: str = Field(..., description="Claims start date in YYYY-MM-DD format.", pattern=r"^\d{4}-\d{2}-\d{2}$")
    CLM_THRU_DT: str = Field(..., description="Claims end date in YYYY-MM-DD format.", pattern=r"^\d{4}-\d{2}-\d{2}$")
    PRVDR_NUM: str = Field(..., description="Provider Institution identifier.")
    ICD9_DGNS_CD_1: Optional[str] = Field(None, description="Claim Diagnosis Code 1.")
    ICD9_DGNS_CD_2: Optional[str] = Field(None, description="Claim Diagnosis Code 2.")
    ICD9_DGNS_CD_3: Optional[str] = Field(None, description="Claim Diagnosis Code 3.")
    ICD9_DGNS_CD_4: Optional[str] = Field(None, description="Claim Diagnosis Code 4.")
    ICD9_DGNS_CD_5: Optional[str] = Field(None, description="Claim Diagnosis Code 5.")
    ICD9_DGNS_CD_6: Optional[str] = Field(None, description="Claim Diagnosis Code 6.")
    ICD9_DGNS_CD_7: Optional[str] = Field(None, description="Claim Diagnosis Code 7.")
    ICD9_DGNS_CD_8: Optional[str] = Field(None, description="Claim Diagnosis Code 8.")

@app.post("/predict", tags=["Prediction"], summary="Predict Admission and Diagnosis Code", description="Predict whether a patient will be admitted based on medical records. If admission is predicted, the Diagnosis Code will also be predicted.")
async def predict(record: Record):
    try:
        record_df = pd.DataFrame([record.model_dump()])
        Claim_ID = record_df['CLM_ID'].iloc[0]
        DESYNPUF_ID = record_df['DESYNPUF_ID'].iloc[0]
        print(Claim_ID)
        print(DESYNPUF_ID)
        record_df.to_csv("Records.csv")
        try:
            preprocessor = DataPreprocessor(record_df, is_training=False)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            print(record_df.shape)
            raise e

        processed_record = preprocessor.data
        print(processed_record)
        processed_record.to_csv('adm_test_record.csv', index=False)
        admission_prediction = admission_model.predict(processed_record)
        print(admission_prediction)

        if int(admission_prediction[0]) == 1:
            processed_record['isAdm'] = 1  
            adm_icd=processed_record.drop(columns=['ADMTNG_ICD9_DGNS_CD_vec_1','ADMTNG_ICD9_DGNS_CD_vec_2','isAdm'])
            adm_icd_prediction = ADM_ICD_model.predict(adm_icd)
            print(adm_icd_prediction)

            Adm_code= embedder.to_code(adm_icd_prediction)
            Category =preprocessor._get_immediate_parent(Adm_code[0])
            return {
                "Claim ID": int(Claim_ID), 
                "Beneficiary_ID": str(DESYNPUF_ID),
                "admission_prediction": int(admission_prediction[0]),
                "Category": Category,
                "Admission_diagnosis_ICD": Adm_code[0]
            }
        else:
            return {
                "Claim ID": int(Claim_ID),  
                "Beneficiary_ID": str(DESYNPUF_ID),
                "admission_prediction": int(admission_prediction[0]),
                "Adm_icd_prediction": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
