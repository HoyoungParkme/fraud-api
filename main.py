from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np

app = FastAPI()

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("fraud_model.h5")
scaler = joblib.load("scaler.pkl")

# 입력 데이터 스키마 정의
class FraudInput(BaseModel):
    features: list  # 예: [1, 0, 3, 1, ...]

@app.post("/predict")
def predict(input: FraudInput):
    X = scaler.transform([input.features])
    prob = model.predict(X)[0][0]
    return {"fraud_prob": float(prob), "fraud": int(prob > 0.5)}
