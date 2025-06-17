import pandas as pd

# 데이터 로드
DATA_PATH = "data/train.csv"
df = pd.read_csv(DATA_PATH)

# ✅ 사용자 입력에 필요한 컬럼들의 고유값 확인
input_columns = [
    'gender', 'marital_status', 'address_change_ind', 'living_status',
    'accident_site', 'witness_present_ind', 'channel',
    'vehicle_category', 'vehicle_color'
]

# ✅ 각 컬럼별 고유값 출력
for col in input_columns:
    print(f"{col}: {sorted(df[col].dropna().unique())}")
