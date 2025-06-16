import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 데이터 로드 (전처리 전 원본 train.csv 파일 경로로 수정 필요)
train_df = pd.read_csv("train.csv")

# 라벨 인코딩할 컬럼
label_encode_columns = [
    'gender', 'marital_status', 'address_change_ind',
    'living_status', 'witness_present_ind', 'age_group',
    'safty_rating_group', 'income_group', 'price_group',
    'weight_group', 'liab_prct_group', 'claim_est_payout_group',
    'age_of_vehicle_group'
]

# binning 먼저 해야 함
train_df['age_group'] = pd.cut(train_df['age_of_driver'], [0, 25, 35, 45, 55, 65, float('inf')],
                               labels=['18-25세', '26-35세', '36-45세', '46-55세', '56-65세', '66세 이상'])
train_df['safty_rating_group'] = pd.cut(train_df['safty_rating'], [0, 20, 40, 60, 80, 100],
                                        labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
train_df['income_group'] = pd.cut(train_df['annual_income'], [0, 35000, 40000, float('inf')],
                                  labels=['0-35000', '35000-40000', '40000+'])
train_df['price_group'] = pd.cut(train_df['vehicle_price'], [0, 20000, 40000, float('inf')],
                                 labels=['0-20000', '20000-40000', '40000+'])
train_df['weight_group'] = pd.cut(train_df['vehicle_weight'], [0, 10000, 20000, float('inf')],
                                  labels=['0-10000', '10000-20000', '20000+'])
train_df['liab_prct_group'] = pd.cut(train_df['liab_prct'], [0, 20, 40, 60, 80, 100],
                                     labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])
train_df['claim_est_payout_group'] = pd.cut(train_df['claim_est_payout'], [0, 5000, 10000, 15000, float('inf')],
                                            labels=['0-5000', '5001-10000', '10001-15000', '15001+'])
train_df['age_of_vehicle_group'] = pd.cut(train_df['age_of_vehicle'], [0, 5, 10, 15, float('inf')],
                                          labels=['0-5년', '6-10년', '11-15년', '16년 이상'])

# 라벨 인코딩하고 각각 저장
for col in label_encode_columns:
    le = LabelEncoder()
    train_df[col] = train_df[col].astype(str)  # 문자열로 변환
    le.fit(train_df[col])
    joblib.dump(le, f'label_encoder_{col}.pkl')
    print(f"Saved: label_encoder_{col}.pkl")
