from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("fraud_model.h5")
scaler = joblib.load("scaler.pkl")

# 라벨 인코더 로드
label_encode_columns = [
    'gender', 'marital_status', 'address_change_ind',
    'living_status', 'witness_present_ind', 'age_group',
    'safty_rating_group', 'income_group', 'price_group',
    'weight_group', 'liab_prct_group', 'claim_est_payout_group',
    'age_of_vehicle_group'
]
label_encoders = {
    col: joblib.load(f"label_encoder_{col}.pkl")
    for col in label_encode_columns
}

# 입력 데이터 스키마 정의
class RawInput(BaseModel):
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
    safty_rating: float
    annual_income: float
    claim_est_payout: float
    age_of_vehicle: float
    vehicle_price: float
    vehicle_weight: float
    liab_prct: float

@app.post("/predict")
def predict(input: RawInput):
    # 데이터프레임 변환
    df = pd.DataFrame([input.dict()])

    # 그룹화 컬럼 생성
    df['age_group'] = pd.cut(df['age_of_driver'], [0, 25, 35, 45, 55, 65, float('inf')],
                             labels=['18-25세', '26-35세', '36-45세', '46-55세', '56-65세', '66세 이상'])
    df['safty_rating_group'] = pd.cut(df['safty_rating'], [0, 20, 40, 60, 80, 100],
                                      labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
    df['income_group'] = pd.cut(df['annual_income'], [0, 35000, 40000, float('inf')],
                                labels=['0-35000', '35000-40000', '40000+'])
    df['price_group'] = pd.cut(df['vehicle_price'], [0, 20000, 40000, float('inf')],
                               labels=['0-20000', '20000-40000', '40000+'])
    df['weight_group'] = pd.cut(df['vehicle_weight'], [0, 10000, 20000, float('inf')],
                                labels=['0-10000', '10000-20000', '20000+'])
    df['liab_prct_group'] = pd.cut(df['liab_prct'], [0, 20, 40, 60, 80, 100],
                                   labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])
    df['claim_est_payout_group'] = pd.cut(df['claim_est_payout'], [0, 5000, 10000, 15000, float('inf')],
                                          labels=['0-5000', '5001-10000', '10001-15000', '15001+'])
    df['age_of_vehicle_group'] = pd.cut(df['age_of_vehicle'], [0, 5, 10, 15, float('inf')],
                                        labels=['0-5년', '6-10년', '11-15년', '16년 이상'])

    # 필요없는 컬럼 제거
    drop_cols = ['age_of_driver', 'safty_rating', 'annual_income', 'vehicle_price',
                 'vehicle_weight', 'claim_est_payout', 'liab_prct', 'age_of_vehicle']
    df.drop(columns=drop_cols, inplace=True)

    # 라벨 인코딩
    for col in label_encode_columns:
        df[col] = label_encoders[col].transform(df[col].astype(str))

    # 원핫 인코딩
    one_hot_cols = ['accident_site', 'channel', 'vehicle_category', 'vehicle_color']
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # 누락된 컬럼 보정 (학습 당시 기준)
    expected_columns = joblib.load("expected_columns.pkl")  # X_train.columns 저장해둔 파일
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]  # 순서 맞춤

    # 스케일링
    X_scaled = scaler.transform(df)

    # 예측
    prob = model.predict(X_scaled)[0][0]
    return {"fraud_prob": float(prob), "fraud": int(prob > 0.5)}
