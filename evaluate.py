import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. CẤU HÌNH ---
DATA_FILE = 'data_onkk.csv'
MODEL_DIR = 'models'
SPLIT_DIR = 'splits'
TEST_DATE_FILE = 'test_dates.txt'

# --- CẤU HÌNH 5 LỚP ---
TARGET_NAMES = ["Tốt (0-50)", "TB (51-100)", "Kém (101-150)", "Xấu (151-200)", "Rất xấu (>200)"]
TARGET_LABELS = [0, 1, 2, 3, 4]

# Hàm đọc ngày từ file txt
def load_dates_from_file(filename):
    path = os.path.join(SPLIT_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file {path}. Sẽ dừng đánh giá.")
        return []
    with open(path, 'r') as f:
        dates = [line.strip() for line in f if line.strip()]
    return dates

# --- 2. CÔNG THỨC TÍNH AQI CHUẨN VIỆT NAM (GIỮ NGUYÊN) ---
def calculate_aqi(pm25):
    """Tính chỉ số AQI chính xác theo công thức"""
    # Bảng 1: Quy định giá trị BP và I 
    breakpoints = [
        (0, 25, 0, 50), (25, 50, 50, 100), (50, 80, 100, 150), 
        (80, 150, 150, 200), (150, 250, 200, 300), 
        (250, 350, 300, 400), (350, 500, 400, 500)
    ]
    Cx = float(pm25) 
    if Cx < 0: return 0
    if Cx > 500: return 500 
    
    for bp in breakpoints:
        BP_lo, BP_hi, I_lo, I_hi = bp
        if BP_lo <= Cx <= BP_hi:
            tu_so = I_hi - I_lo
            mau_so = BP_hi - BP_lo
            hieu_so = Cx - BP_lo
            aqi = (tu_so / mau_so) * hieu_so + I_lo
            return aqi
    return 0

def get_aqi_category_from_index(aqi_val):
    if aqi_val <= 50: return 0 
    elif aqi_val <= 100: return 1
    elif aqi_val <= 150: return 2
    elif aqi_val <= 200: return 3
    else: return 4

def pm25_to_label_final(pm25):
    aqi_index = calculate_aqi(pm25)
    return get_aqi_category_from_index(aqi_index)

# --- 3. CHUẨN BỊ VÀ LỌC DỮ LIỆU ---
print("Đang đọc dữ liệu...")
df = pd.read_csv(DATA_FILE)
df['time'] = pd.to_datetime(df['time'])

# Load danh sách ngày Test
test_dates_list = load_dates_from_file(TEST_DATE_FILE)
if not test_dates_list:
    exit()

test_dt = pd.to_datetime(test_dates_list)

# Lọc: Chỉ giữ lại các dòng thuộc ngày Test
eval_df = df[df['time'].isin(test_dt)].copy()

if len(eval_df) == 0:
    print(f"❌ LỖI: Không tìm thấy mẫu dữ liệu nào trong file {DATA_FILE} khớp với ngày trong {TEST_DATE_FILE}.")
    exit()

# LƯU Ý: Nếu bạn đã thêm Feature Engineering (như Stagnation/DewPoint) ở file train, 
# BẠN CŨNG CẦN THÊM VÀO ĐÂY trước khi định nghĩa features.
# Ví dụ: eval_df['Stagnation'] = 1 / (eval_df['WSPD'] + 0.1)

features = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT'] 
X_test = eval_df[features]

# TẠO NHÃN THẬT (GROUND TRUTH)
print(f"Đang tính toán nhãn thực tế từ số liệu trạm ({len(eval_df)} mẫu Test)...")
y_true = eval_df['pm25'].apply(pm25_to_label_final).values

# 4. ĐÁNH GIÁ MODEL
models_to_test = ['rf_reg_phys.pkl', 'xgb_reg_phys.pkl'] # Tên model sau khi tối ưu

print(f"\n{'='*10} BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP {'='*10}")

for model_name in models_to_test:
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Kiểm tra các phiên bản model cũ nếu không tìm thấy tên model mới
    if not os.path.exists(model_path):
        model_name = model_name.replace('_phys', '')
        model_path = os.path.join(MODEL_DIR, model_name)
        
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file: {model_name}")
        continue
        
    print(f"\n🔍 Đang đánh giá: {model_name}...")
    model = joblib.load(model_path)
    
    # A. Dự báo ra nồng độ PM2.5 (Số thực)
    y_pred_pm25 = model.predict(X_test)
    
    # B. Tính AQI từ PM2.5 dự báo -> Quy ra nhãn
    y_pred_class = [pm25_to_label_final(val) for val in y_pred_pm25]
    
    # --- TÍNH TOÁN CHỈ SỐ ---
    acc = accuracy_score(y_true, y_pred_class)
    print(f"Độ chính xác phân lớp (Accuracy): {acc:.2%}")
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_true, 
        y_pred_class, 
        labels=TARGET_LABELS,      
        target_names=TARGET_NAMES,  
        zero_division=0
    ))
    
    # VẼ CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred_class, labels=TARGET_LABELS)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGET_NAMES, 
                yticklabels=TARGET_NAMES)
    plt.xlabel('Dự báo (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'Confusion Matrix - {model_name}\n')
    plt.show()
    
    print("-" * 50)