from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# FastAPI 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 전처리기 로드
MODEL_DIR = "model_assets"
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "fraud_model.h5"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
expected_columns = joblib.load(os.path.join(MODEL_DIR, "expected_columns.pkl"))

# 입력 데이터 스키마 정의
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
    # 입력값을 DataFrame으로 변환
    input_df = pd.DataFrame([data.dict()])

    # 파생 변수 생성
    input_df["age_group"] = pd.cut(
        input_df["age_of_driver"], [0, 25, 35, 45, 55, 65, float("inf")],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    )
    input_df["safty_rating_group"] = pd.cut(
        input_df["safty_rating"], [0, 20, 40, 60, 80, 100],
        labels=["0-20", "21-40", "41-60", "61-80", "81-100"]
    )
    input_df["income_group"] = pd.cut(
        input_df["annual_income"], [0, 35000, 40000, float("inf")],
        labels=["0-35000", "35000-40000", "40000+"]
    )
    input_df["price_group"] = pd.cut(
        input_df["vehicle_price"], [0, 20000, 40000, float("inf")],
        labels=["0-20k", "20k-40k", "40k+"]
    )
    input_df["weight_group"] = pd.cut(
        input_df["vehicle_weight"], [0, 10000, 20000, float("inf")],
        labels=["0-10t", "10t-20t", "20t+"]
    )
    input_df["liab_prct_group"] = pd.cut(
        input_df["liab_prct"], [0, 20, 40, 60, 80, 100],
        labels=["0-20", "21-40", "41-60", "61-80", "81-100"]
    )
    input_df["claim_est_payout_group"] = pd.cut(
        input_df["claim_est_payout"], [0, 5000, 10000, 15000, float("inf")],
        labels=["0-5k", "5k-10k", "10k-15k", "15k+"]
    )
    input_df["age_of_vehicle_group"] = pd.cut(
        input_df["age_of_vehicle"], [0, 5, 10, 15, float("inf")],
        labels=["0-5", "5-10", "10-15", "15+"]
    )

    # 파생 변수 적용 후 원래 수치형 컬럼 제거
    drop_cols = [
        "age_of_driver", "safty_rating", "annual_income",
        "vehicle_price", "vehicle_weight", "liab_prct",
        "claim_est_payout", "age_of_vehicle"
    ]
    input_df = input_df.drop(columns=drop_cols)

    # 라벨 인코딩
    for col in input_df.columns:
        le_path = os.path.join(MODEL_DIR, f"label_encoder_{col}.pkl")
        if os.path.exists(le_path):
            le = joblib.load(le_path)
            input_df[col] = le.transform(input_df[col].astype(str))

    # 원핫 인코딩
    input_df = pd.get_dummies(input_df, drop_first=False)

    # 누락된 컬럼 채우기
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # 스케일링
    input_scaled = scaler.transform(input_df)

    # 예측
    pred = model.predict(input_scaled)
    probability = float(pred[0][0])
    is_fraud = probability > 0.5

    return {
        "probability": round(probability, 4),
        "is_fraud": is_fraud
    }
