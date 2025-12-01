import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. CẤU HÌNH ---
DATA_FILE = 'data_onkk.csv'
MODEL_DIR = 'models'
SPLIT_DIR = 'splits'  # Thư mục chứa file dates

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
np.random.seed(2) # Giữ nguyên seed cũ của bạn

# Hàm đọc ngày từ file txt (MỚI)
def load_dates_from_file(filename):
    path = os.path.join(SPLIT_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file {path}. Hãy kiểm tra lại!")
        return []
    with open(path, 'r') as f:
        dates = [line.strip() for line in f if line.strip()]
    return dates

print("🔄 Đang xử lý dữ liệu...")
df = pd.read_csv(DATA_FILE)
df['time'] = pd.to_datetime(df['time'])

# --- 2. TÁCH TRAIN / TEST (SỬA ĐỔI) ---
# Thay vì hardcode, ta đọc từ file
test_dates_list = load_dates_from_file('test_dates.txt')
test_dt = pd.to_datetime(test_dates_list)

if len(test_dt) == 0:
    print("❌ LỖI: Danh sách ngày test trống. Chương trình dừng lại.")
    exit()

# Test: Những ngày có trong file test_dates.txt
test_df = df[df['time'].isin(test_dt)]

# Train: TẤT CẢ những ngày còn lại (bao gồm cả val cũ nếu có, dồn hết vào train)
train_df = df[~df['time'].isin(test_dt)]

print(f"\n📊 Thống kê dữ liệu:")
print(f"   - Train (Học): {len(train_df)} dòng")
print(f"   - Test (Chấm điểm): {len(test_df)} dòng")

# --- 3. TĂNG CƯỜNG DỮ LIỆU THÔNG MINH (GIỮ NGUYÊN) ---
def augment_data(data, n_copies=1, noise_level=0.02):
    """
    Tạo dữ liệu giả lập bằng cách thêm nhiễu Gaussian vào các biến khí tượng.
    """
    augmented = []
    features_dynamic = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']
    
    for _ in range(n_copies):
        new_data = data.copy()
        for col in features_dynamic:
            std_dev = new_data[col].std()
            noise = np.random.normal(0, std_dev * noise_level, size=len(new_data))
            new_data[col] = new_data[col] + noise
        augmented.append(new_data)
        
    return pd.concat(augmented, axis=0)

# Lọc dữ liệu cực trị
high_pollution = train_df[train_df['pm25'] > 80] # Ô nhiễm
clean_air = train_df[train_df['pm25'] < 20]      # Sạch

# Tạo dữ liệu biến dị
high_aug = augment_data(high_pollution, n_copies=2, noise_level=0.03) 
clean_aug = augment_data(clean_air, n_copies=1, noise_level=0.03)

# Gộp lại: Gốc + Biến dị
train_final = pd.concat([train_df, high_aug, clean_aug], axis=0)

print(f"   - Train sau khi tăng cường: {len(train_final)} dòng")

features = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT']
X_train = train_final[features]
y_train = train_final['pm25']

X_test = test_df[features]
y_test = test_df['pm25']

# --- 4. HUẤN LUYỆN (GIỮ NGUYÊN) ---
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

print(f"\n{'='*10} TRAIN MODEL (OLD CONFIG) {'='*10}")

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
    print(f"   Đã lưu model vào {MODEL_DIR}/{filename}.\n")