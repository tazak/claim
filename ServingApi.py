import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from DataPreprocessor import DataPreprocessor
import joblib

app = FastAPI()

admission_model_path = "xgb_model_admission.pkl"  
category_model_path = "xgb_model_category.pkl"  
label_encoders_path = "label_encoders.pkl"  

try:
    global admission_model, category_model
    admission_model = joblib.load(admission_model_path)
    category_model = joblib.load(category_model_path)
    label_encoders = joblib.load(label_encoders_path)
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")


# Pydantic model for request validation
class Record(BaseModel):
    DESYNPUF_ID: str
    BENE_BIRTH_DT: str
    BENE_DEATH_DT: Optional[str] = None
    BENE_SEX_IDENT_CD: int
    BENE_RACE_CD: int
    BENE_ESRD_IND: str
    BENE_HI_CVRAGE_TOT_MONS: int
    BENE_SMI_CVRAGE_TOT_MONS: int
    BENE_HMO_CVRAGE_TOT_MONS: int
    PLAN_CVRG_MOS_NUM: int
    SP_ALZHDMTA: int
    SP_CHF: int
    SP_CHRNKIDN: int
    SP_CNCR: int
    SP_COPD: int
    SP_DEPRESSN: int
    SP_DIABETES: int
    SP_ISCHMCHT: int
    SP_OSTEOPRS: int
    SP_RA_OA: int
    SP_STRKETIA: int
    MEDREIMB_IP: int
    BENRES_IP: int
    PPPYMT_IP: int
    MEDREIMB_OP: int
    BENRES_OP: int
    PPPYMT_OP: int
    MEDREIMB_CAR: int
    BENRES_CAR: int
    PPPYMT_CAR: int
    CLM_ID: int
    CLM_FROM_DT: str
    CLM_THRU_DT: str
    PRVDR_NUM: str
    ICD9_DGNS_CD_1: Optional[str] = None
    ICD9_DGNS_CD_2: Optional[str] = None
    ICD9_DGNS_CD_3: Optional[str] = None
    ICD9_DGNS_CD_4: Optional[str] = None
    ICD9_DGNS_CD_5: Optional[str] = None
    ICD9_DGNS_CD_6: Optional[str] = None
    ICD9_DGNS_CD_7: Optional[str] = None
    ICD9_DGNS_CD_8: Optional[str] = None

@app.post("/predict")
async def predict(record: Record):
    try:
        record_df = pd.DataFrame([record.model_dump()])

        preprocessor = DataPreprocessor(record_df, isTraining=False)
        processed_record = preprocessor.data
        admission_prediction = admission_model.predict(processed_record)

        return {
            "admission_prediction": int(admission_prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
