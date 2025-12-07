import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# ==============================================================================
# 1. HÀM TÍNH AQI VÀ CHIA NHÓM (Dựa trên nồng độ PM2.5)
# ==============================================================================
def get_aqi_group(pm25):
    # Sử dụng các điểm cắt (breakpoints) phổ biến của US EPA cho PM2.5 (µg/m3)
    # Nhóm 0: Tốt (0 - 12.0)
    # Nhóm 1: Trung bình (12.1 - 35.4)
    # Nhóm 2: Kém cho người nhạy cảm (35.5 - 55.4)
    # Nhóm 3: Xấu (55.5 - 150.4)
    # Nhóm 4: Rất xấu (150.5 - 250.4) và Nguy hại (> 250.5)
    
    if pm25 <= 12.0: return 0
    elif pm25 <= 35.4: return 1
    elif pm25 <= 55.4: return 2
    elif pm25 <= 150.4: return 3
    else: return 4 # Bao gồm cả Rất xấu và Nguy hại (do dữ liệu max ~213)

# ==============================================================================
# 2. LOAD DATA & FEATURE ENGINEERING
# ==============================================================================
df = pd.read_csv('data_onkk.csv')

# Xử lý thời gian
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear
df['day_of_week'] = df['time'].dt.dayofweek

# Tạo cột Nhóm AQI để dùng cho SMOTE
df['aqi_group'] = df['pm25'].apply(get_aqi_group)

# Xem phân bố các nhóm trước khi SMOTE
print("Phân bố dữ liệu theo nhóm AQI gốc:")
print(df['aqi_group'].value_counts().sort_index())

# Loại bỏ các cột không train
df_model = df.drop(columns=['time', 'ID'])

# Tách Features (X) và Target (y)
X = df_model.drop(columns=['pm25', 'aqi_group']) # X chưa có pm25
y = df_model['pm25']
groups = df_model['aqi_group'] # Chỉ dùng để SMOTE

# ==============================================================================
# 3. CHIA TRAIN / TEST (80/20)
# ==============================================================================
# Lưu ý: Stratify theo nhóm AQI để đảm bảo tập Test cũng đủ đại diện các nhóm
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=groups
)

# ==============================================================================
# 4. ÁP DỤNG SMOTE CHO BÀI TOÁN HỒI QUY (Regression)
# ==============================================================================
# Kỹ thuật: Đưa y (pm25) vào X tạm thời, SMOTE theo nhóm, rồi tách ra lại.

print("\nĐang thực hiện SMOTE trên tập Train...")

# 4.1. Gộp X_train và y_train tạm thời
X_train_with_target = X_train.copy()
X_train_with_target['TARGET_PM25'] = y_train

# 4.2. Áp dụng SMOTE để cân bằng số lượng mẫu giữa các nhóm AQI (g_train)
# k_neighbors=3 vì các nhóm hiếm (nhóm 4) có thể có ít mẫu
smote = SMOTE(random_state=42, k_neighbors=3) 
X_res, g_res = smote.fit_resample(X_train_with_target, g_train)

# 4.3. Tách lại X và y sau khi SMOTE
y_train_res = X_res['TARGET_PM25']
X_train_res = X_res.drop(columns=['TARGET_PM25'])

print(f"Kích thước Train gốc: {X_train.shape[0]}, Sau SMOTE: {X_train_res.shape[0]}")

# ==============================================================================
# 5. TRAINING VỚI K-FOLD CROSS VALIDATION
# ==============================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Random Forest ---
rf = RandomForestRegressor(random_state=42)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2]
}
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=10, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

print("\nĐang tối ưu Random Forest...")
rf_search.fit(X_train_res, y_train_res) # Train trên tập đã SMOTE
best_rf = rf_search.best_estimator_

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_params = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}
xgb_search = RandomizedSearchCV(xgb_model, xgb_params, n_iter=10, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

print("Đang tối ưu XGBoost...")
xgb_search.fit(X_train_res, y_train_res) # Train trên tập đã SMOTE
best_xgb = xgb_search.best_estimator_

# ==============================================================================
# 6. EVALUATE (ĐÁNH GIÁ TRÊN FILE TEST GỐC)
# ==============================================================================
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Tính thêm độ chính xác theo phân loại AQI (Optional)
    pred_groups = [get_aqi_group(p) for p in y_pred]
    true_groups = [get_aqi_group(p) for p in y_test]
    accuracy_aqi = np.mean(np.array(pred_groups) == np.array(true_groups))
    
    print(f"\n--- {name} Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"AQI Group Accuracy: {accuracy_aqi*100:.2f}% (Tỉ lệ dự báo đúng nhóm màu)")

evaluate(best_rf, X_test, y_test, "Random Forest")
evaluate(best_xgb, X_test, y_test, "XGBoost")

# ==============================================================================
# 7. HÀM DỰ BÁO (PREDICT)
# ==============================================================================
def predict_new_pixel(features_dict):
    # features_dict: {'lat': ..., 'lon': ..., 'PRES2M': ...}
    df_input = pd.DataFrame([features_dict])
    # Đảm bảo thứ tự cột giống lúc train
    df_input = df_input[X.columns] 
    return best_xgb.predict(df_input)[0]

def save_artifacts():
    # Tạo thư mục 'models' nếu chưa có
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print("Đang lưu models và tập Test vào thư mục 'models/'...")
    
    # 1. Lưu Models đã train xong
    joblib.dump(best_rf, 'models/best_rf_model.pkl')
    joblib.dump(best_xgb, 'models/best_xgb_model.pkl')
    
    # 2. LƯU TẬP TEST (Quan trọng: Để file evaluate riêng có dữ liệu mà test)
    # Ta cần lưu cả X_test và y_test
    joblib.dump(X_test, 'models/X_test.pkl')
    joblib.dump(y_test, 'models/y_test.pkl')
    

print(f"\nBest RF Params: {rf_search.best_params_}")
print(f"Best XGB Params: {xgb_search.best_params_}")
save_artifacts()