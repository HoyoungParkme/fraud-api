### main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# FastAPI 앱 설정
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 도구 로드
model = tf.keras.models.load_model("model_assets/fraud_model.h5")
scaler = joblib.load("model_assets/scaler.pkl")
expected_columns = joblib.load("model_assets/expected_columns.pkl")

# 라벨 인코더 로딩
def load_label_encoders():
    label_cols = [
        'gender', 'marital_status', 'address_change_ind', 'living_status',
        'witness_present_ind', 'age_group', 'safty_rating_group', 'income_group',
        'price_group', 'weight_group', 'liab_prct_group', 'claim_est_payout_group',
        'age_of_vehicle_group'
    ]
    encoders = {}
    for col in label_cols:
        encoders[col] = joblib.load(f"model_assets/label_encoder_{col}.pkl")
    return encoders

label_encoders = load_label_encoders()

# 입력 데이터 스키마
class FraudInput(BaseModel):
    gender: str
    marital_status: str
    high_education_ind: int
    address_change_ind: str
    living_status: str
    accident_site: str
    past_num_of_claims: int
    witness_present_ind: str
    policy_report_filed_ind: int
    channel: str
    vehicle_category: str
    vehicle_color: str
    age_of_driver: int
    safty_rating: int
    annual_income: float
    claim_est_payout: float
    age_of_vehicle: int
    vehicle_price: float
    vehicle_weight: float
    liab_prct: int

@app.post("/predict")
def predict(data: FraudInput):
    try:
        # 입력 데이터 프레임 변환
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # 파생 변수 생성
        df["age_group"] = pd.cut(df["age_of_driver"], [0, 25, 35, 45, 55, 65, float("inf")], labels=[0, 1, 2, 3, 4, 5])
        df["safty_rating_group"] = pd.cut(df["safty_rating"], [0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4])
        df["income_group"] = pd.cut(df["annual_income"], [0, 35000, 40000, float("inf")], labels=[0, 1, 2])
        df["price_group"] = pd.cut(df["vehicle_price"], [0, 20000, 40000, float("inf")], labels=[0, 1, 2])
        df["weight_group"] = pd.cut(df["vehicle_weight"], [0, 10000, 20000, float("inf")], labels=[0, 1, 2])
        df["liab_prct_group"] = pd.cut(df["liab_prct"], [0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4])
        df["claim_est_payout_group"] = pd.cut(df["claim_est_payout"], [0, 5000, 10000, 15000, float("inf")], labels=[0, 1, 2, 3])
        df["age_of_vehicle_group"] = pd.cut(df["age_of_vehicle"], [0, 5, 10, 15, float("inf")], labels=[0, 1, 2, 3])

        # 인코딩
        for col, encoder in label_encoders.items():
            df[col] = encoder.transform(df[col].astype(str))

        # 불필요한 컬럼 제거
        df.drop(columns=[
            "age_of_driver", "safty_rating", "annual_income",
            "vehicle_price", "vehicle_weight", "liab_prct",
            "claim_est_payout", "age_of_vehicle"
        ], inplace=True)

        # 원핫 인코딩
        df = pd.get_dummies(df, columns=["accident_site", "channel", "vehicle_category", "vehicle_color"], drop_first=False)

        # 누락된 컬럼 추가
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]  # 컬럼 순서 맞추기

        # 스케일링 및 예측
        X_scaled = scaler.transform(df)
        pred = model.predict(X_scaled)[0][0]
        return {"prediction": float(pred)}

    except Exception as e:
        return {"error": str(e)}
