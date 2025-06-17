### test_predict_local.py

import requests

url = "http://localhost:8000/predict"

sample = {
    "gender": "Male",
    "marital_status": "Married",
    "high_education_ind": 0,
    "address_change_ind": "Changed",
    "living_status": "Own",
    "accident_site": "Local",
    "past_num_of_claims": 0,
    "witness_present_ind": "No witness",
    "policy_report_filed_ind": 1,
    "channel": "Broker",
    "vehicle_category": "Compact",
    "vehicle_color": "black",
    "age_of_driver": 33,
    "safty_rating": 34,
    "annual_income": 35113.78,
    "claim_est_payout": 2748.61,
    "age_of_vehicle": 8,
    "vehicle_price": 19799.63,
    "vehicle_weight": 11640.45,
    "liab_prct": 42
}

res = requests.post(url, json=sample)
print("== Response ==")
print(res.status_code)
print(res.json())