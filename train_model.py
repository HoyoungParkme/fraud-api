import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report

# 경로 설정
DATA_PATH = "data/train.csv"
MODEL_DIR = "model_assets"
os.makedirs(MODEL_DIR, exist_ok=True)

# 데이터 로드
df = pd.read_csv(DATA_PATH)
y = df["fraud"]

# 파생 변수 생성
X = df.copy()
X["age_group"] = pd.cut(df["age_of_driver"], [0, 25, 35, 45, 55, 65, float("inf")], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
X["safty_rating_group"] = pd.cut(df["safty_rating"], [0, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
X["income_group"] = pd.cut(df["annual_income"], [0, 35000, 40000, float("inf")], labels=["0-35000", "35000-40000", "40000+"])
X["price_group"] = pd.cut(df["vehicle_price"], [0, 20000, 40000, float("inf")], labels=["0-20k", "20k-40k", "40k+"])
X["weight_group"] = pd.cut(df["vehicle_weight"], [0, 10000, 20000, float("inf")], labels=["0-10t", "10t-20t", "20t+"])
X["liab_prct_group"] = pd.cut(df["liab_prct"], [0, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
X["claim_est_payout_group"] = pd.cut(df["claim_est_payout"], [0, 5000, 10000, 15000, float("inf")], labels=["0-5k", "5k-10k", "10k-15k", "15k+"])
X["age_of_vehicle_group"] = pd.cut(df["age_of_vehicle"], [0, 5, 10, 15, float("inf")], labels=["0-5", "5-10", "10-15", "15+"])

# 학습에 불필요한 컬럼 제거
X = X.drop(columns=["ID", "fraud", "age_of_driver", "safty_rating", "annual_income", "vehicle_price", "vehicle_weight", "year", "month", "day", "claim_day_of_week"])

# 라벨 인코딩
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    joblib.dump(le, os.path.join(MODEL_DIR, f"label_encoder_{col}.pkl"))
    label_encoders[col] = le

# 원핫 인코딩
X = pd.get_dummies(X, drop_first=False)
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "expected_columns.pkl"))

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# 불균형 데이터 보정 함수
def resample_data(X, y, method="smoteenn"):
    if method == "smoteenn":
        sampler = SMOTEENN(random_state=42)
    else:
        raise ValueError("지원하지 않는 샘플링 방식")
    return sampler.fit_resample(X, y)

# 리샘플링 적용
X_resampled, y_resampled = resample_data(X_scaled, y, method="smoteenn")

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=1)

# 성능 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, digits=2))

# 모델 저장
model.save(os.path.join(MODEL_DIR, "fraud_model.h5"))
