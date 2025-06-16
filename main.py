from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중엔 "*" 가능. 배포 후엔 출처 지정 추천
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("fraud_model.h5")
scaler = joblib.load("scaler.pkl")

# 입력 스키마
class FraudInput(BaseModel):
    features: list

@app.post("/predict")
def predict(input: FraudInput):
    X = scaler.transform([input.features])
    prob = model.predict(X)[0][0]
    return {"fraud_prob": float(prob), "fraud": int(prob > 0.5)}
