import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Cấu hình
DATA_FILE = 'data_onkk.csv'
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

print("⏳ Đang đọc và xử lý dữ liệu...")
df = pd.read_csv(DATA_FILE)

# --- CHIẾN LƯỢC 1: TĂNG CƯỜNG DỮ LIỆU (OVERSAMPLING) ---
# Lọc ra những ngày ô nhiễm nặng (PM2.5 > 80 - Mức Kém trở lên)
high_pollution = df[df['pm25'] > 80]
# Lọc ra những ngày rất trong lành (PM2.5 < 20) để cân bằng
clean_air = df[df['pm25'] < 20]

# Nhân bản dữ liệu quan trọng lên (3 lần cho ô nhiễm, 2 lần cho trong lành)
# Điều này buộc model phải học kỹ các trường hợp này thay vì chỉ học cái trung bình
df_balanced = pd.concat([df, high_pollution, high_pollution, high_pollution, clean_air], axis=0)

print(f"   -> Dữ liệu gốc: {len(df)} dòng")
print(f"   -> Dữ liệu sau khi tăng cường: {len(df_balanced)} dòng")

features = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT']
X = df_balanced[features]
y = df_balanced['pm25']

# --- CHIẾN LƯỢC 2: CẤU HÌNH MẠNH MẼ ---
models = {
    "rf_reg.pkl": RandomForestRegressor(
        n_estimators=200,       # Nhiều cây hơn
        max_depth=25,           # Cho phép cây mọc sâu để bắt chi tiết nhỏ
        min_samples_leaf=2,     # Giảm nhiễu
        random_state=42, 
        n_jobs=-1
    ),
    "xgb_reg.pkl": XGBRegressor(
        n_estimators=1000,      # Tăng số lượng cây để học kỹ
        learning_rate=0.05,     # Học chậm nhưng chắc (giảm từ 0.1 xuống 0.05)
        max_depth=10,           # Đủ sâu để vẽ biên giới phức tạp (địa hình/thời tiết)
        subsample=0.8,          # Tránh học vẹt (Overfitting)
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1
    )
}

print(f"{'='*10} TRAIN MODEL NÂNG CAO {'='*10}")

for filename, model in models.items():
    print(f"🛠️ Đang train {filename} (Cấu hình mạnh)...")
    model.fit(X, y)
    
    # Đánh giá trên dữ liệu gốc (df) để khách quan, không dùng df_balanced để test
    y_pred = model.predict(df[features])
    r2 = r2_score(df['pm25'], y_pred)
    mse = mean_squared_error(df['pm25'], y_pred)
    rmse = np.sqrt(mse)
    
    print(f"   -> R2 Score (trên dữ liệu gốc): {r2:.4f}")
    print(f"   -> RMSE: {rmse:.2f}")
    
    # Lưu file
    save_path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, save_path)
    print(f"   ✅ Đã lưu model.")

print("\n🎉 Xong! Hãy chạy lại 'streamlit run app_ml.py' để xem bản đồ mới.")