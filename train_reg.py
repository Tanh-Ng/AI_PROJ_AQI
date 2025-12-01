import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. CẤU HÌNH
DATA_FILE = 'data_onkk.csv'
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
np.random.seed(2)

EXCLUDE_DATES = [
    "2020-01-01", "2020-05-29", "2020-07-21",
    "2020-08-10", "2020-10-15", "2020-12-15"
]

print(" Đang xử lý dữ liệu...")
df = pd.read_csv(DATA_FILE)
df['time'] = pd.to_datetime(df['time'])
exclude_dt = pd.to_datetime(EXCLUDE_DATES)

# --- 2. TÁCH TRAIN / TEST ---
test_df = df[df['time'].isin(exclude_dt)]
train_df = df[~df['time'].isin(exclude_dt)]

print(f"    Thống kê:")
print(f"   - Train gốc: {len(train_df)} dòng")
print(f"   - Test: {len(test_df)} dòng")

# --- 3. TĂNG CƯỜNG DỮ LIỆU THÔNG MINH (NOISE INJECTION) ---
def augment_data(data, n_copies=1, noise_level=0.02):
    """
    Tạo dữ liệu giả lập bằng cách thêm nhiễu Gaussian vào các biến khí tượng.
    noise_level=0.02 nghĩa là dao động khoảng 2% so với độ lệch chuẩn.
    """
    augmented = []
    # Chỉ thêm nhiễu vào các biến khí tượng biến đổi, giữ nguyên địa hình
    features_dynamic = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']
    
    for _ in range(n_copies):
        new_data = data.copy()
        for col in features_dynamic:
            # Tạo nhiễu ngẫu nhiên dựa trên độ biến động của cột đó
            std_dev = new_data[col].std()
            noise = np.random.normal(0, std_dev * noise_level, size=len(new_data))
            new_data[col] = new_data[col] + noise
        augmented.append(new_data)
        
    return pd.concat(augmented, axis=0)

# Lọc dữ liệu cực trị
high_pollution = train_df[train_df['pm25'] > 80] # Ô nhiễm
clean_air = train_df[train_df['pm25'] < 20]      # Sạch

# Tạo dữ liệu biến dị (Synthetic Data)
# Tạo ra các bản sao có sai số nhẹ
high_aug = augment_data(high_pollution, n_copies=2, noise_level=0.03) 
clean_aug = augment_data(clean_air, n_copies=1, noise_level=0.03)

# Gộp lại: Gốc + Biến dị
train_final = pd.concat([train_df, high_aug, clean_aug], axis=0)

print(f"   - Train sau khi tăng cường (có nhiễu): {len(train_final)} dòng")

features = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT']
X_train = train_final[features]
y_train = train_final['pm25']

X_test = test_df[features]
y_test = test_df['pm25']

# --- 4. HUẤN LUYỆN ---
models = {
    "rf_reg.pkl": RandomForestRegressor(
        n_estimators=200, 
        max_depth=20,       
        min_samples_leaf=4, 
        random_state=42, n_jobs=-1
    ),
    "xgb_reg.pkl": XGBRegressor(
        n_estimators=800,      
        learning_rate=0.03,     
        max_depth=6,           
        subsample=0.7,          # Chỉ dùng 70% dữ liệu mỗi lần
        colsample_bytree=0.7,   # Chỉ dùng 70% đặc trưng mỗi lần
        random_state=42, n_jobs=-1
    )
}

print(f"\n{'='*10} TRAIN MODEL (SMART AUGMENTATION) {'='*10}")

for filename, model in models.items():
    print(f"Đang train {filename}...")
    model.fit(X_train, y_train)
    
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"       Kết quả Test ({len(X_test)} mẫu):")
        print(f"      -> R2 Score: {r2:.4f}")
        print(f"      -> RMSE: {rmse:.2f}")
    
    joblib.dump(model, os.path.join(MODEL_DIR, filename))
    print(f"   Đã lưu model.")