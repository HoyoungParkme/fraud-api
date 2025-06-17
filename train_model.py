import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 경로 설정
DATA_PATH = "data/train.csv"          # 🚨 train.csv 위치
MODEL_DIR = "model_assets"
os.makedirs(MODEL_DIR, exist_ok=True)

# 데이터 로드
df = pd.read_csv(DATA_PATH)

# 타겟 분리
y = df["fraud"]
X = df.drop(columns=[
    "ID", "fraud", "age_of_driver", "safty_rating", "annual_income",
    "vehicle_price", "vehicle_weight", "year", "month", "day", "claim_day_of_week"
])

# 파생 피처 생성 (삭제된 컬럼 기준으로 원본 df에서 생성)
X["age_group"] = pd.cut(df["age_of_driver"], [0, 25, 35, 45, 55, 65, float("inf")], labels=[0, 1, 2, 3, 4, 5])
X["safty_rating_group"] = pd.cut(df["safty_rating"], [0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4])
X["income_group"] = pd.cut(df["annual_income"], [0, 35000, 40000, float("inf")], labels=[0, 1, 2])
X["price_group"] = pd.cut(df["vehicle_price"], [0, 20000, 40000, float("inf")], labels=[0, 1, 2])
X["weight_group"] = pd.cut(df["vehicle_weight"], [0, 10000, 20000, float("inf")], labels=[0, 1, 2])
X["liab_prct_group"] = pd.cut(df["liab_prct"], [0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4])
X["claim_est_payout_group"] = pd.cut(df["claim_est_payout"], [0, 5000, 10000, 15000, float("inf")], labels=[0, 1, 2, 3])
X["age_of_vehicle_group"] = pd.cut(df["age_of_vehicle"], [0, 5, 10, 15, float("inf")], labels=[0, 1, 2, 3])

# 라벨 인코딩
label_cols = [
    'gender', 'marital_status', 'address_change_ind', 'living_status',
    'witness_present_ind', 'age_group', 'safty_rating_group', 'income_group',
    'price_group', 'weight_group', 'liab_prct_group', 'claim_est_payout_group',
    'age_of_vehicle_group'
]
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    joblib.dump(le, os.path.join(MODEL_DIR, f"label_encoder_{col}.pkl"))
    label_encoders[col] = le

    # 클래스 목록 출력 (디버깅용)
    print(f"{col} 클래스 목록: {list(le.classes_)}")

# 원핫 인코딩
X = pd.get_dummies(X, columns=["accident_site", "channel", "vehicle_category", "vehicle_color"], drop_first=False)

# 컬럼 순서 저장
expected_columns = X.columns.tolist()
joblib.dump(expected_columns, os.path.join(MODEL_DIR, "expected_columns.pkl"))

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# 평가
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {macro_f1}")

# 모델 저장
model.save(os.path.join(MODEL_DIR, "fraud_model.h5"))
