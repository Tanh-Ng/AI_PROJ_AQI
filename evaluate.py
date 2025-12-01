import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. CẤU HÌNH
DATA_FILE = 'data_onkk.csv'
MODEL_DIR = 'models'

# --- CẤU HÌNH 5 LỚP (GỘP NGUY HẠI VÀO RẤT XẤU) ---
# Vì dữ liệu rất ít khi >300
TARGET_NAMES = ["Tốt (0-50)", "TB (51-100)", "Kém (101-150)", "Xấu (151-200)", "Rất xấu (>200)"]
TARGET_LABELS = [0, 1, 2, 3, 4]

# 2. CÔNG THỨC TÍNH AQI CHUẨN VIỆT NAM
def calculate_aqi(pm25):
    """
    Tính chỉ số AQI chính xác theo công thức
    AQI = [ (I_hi - I_lo) / (BP_hi - BP_lo) ] * (Cx - BP_lo) + I_lo
    """
    # Bảng 1: Quy định giá trị BP và I 
    # Cấu trúc: (BP_lo, BP_hi, I_lo, I_hi)
    breakpoints = [
        (0, 25, 0, 50),         # Mức 1
        (25, 50, 50, 100),      # Mức 2
        (50, 80, 100, 150),     # Mức 3
        (80, 150, 150, 200),    # Mức 4
        (150, 250, 200, 300),   # Mức 5
        (250, 350, 300, 400),   # Mức 6
        (350, 500, 400, 500)    # Mức 7
    ]
    
    Cx = float(pm25) # Nồng độ đầu vào (C_x)
    
    # Xử lý ngoại lệ (Ngoài khoảng đo)
    if Cx < 0: return 0
    if Cx > 500: return 500 # Kịch kim bảng tra
    
    for bp in breakpoints:
        BP_lo, BP_hi, I_lo, I_hi = bp
        
        # Kiểm tra xem Cx thuộc khoảng nào [BP_i, BP_i+1]
        if BP_lo <= Cx <= BP_hi:
            # Áp dụng ĐÚNG công thức trong ảnh:
            # (I_i+1 - I_i)
            tu_so = I_hi - I_lo
            
            # (BP_i+1 - BP_i)
            mau_so = BP_hi - BP_lo
            
            # (Cx - BP_i)
            hieu_so = Cx - BP_lo
            
            # Công thức tổng quát
            aqi = (tu_so / mau_so) * hieu_so + I_lo
            
            return aqi
            
    return 0

def get_aqi_category_from_index(aqi_val):
    """Quy đổi từ chỉ số AQI số học sang Class ID (0-4)"""
    if aqi_val <= 50: return 0   # Tốt
    elif aqi_val <= 100: return 1 # Trung bình
    elif aqi_val <= 150: return 2 # Kém
    elif aqi_val <= 200: return 3 # Xấu
    else: return 4               # Rất xấu + Nguy hại (Gộp chung >200)

# Hàm wrapper kết hợp cả 2 bước trên
def pm25_to_label_final(pm25):
    aqi_index = calculate_aqi(pm25)
    return get_aqi_category_from_index(aqi_index)

# 3. CHUẨN BỊ DỮ LIỆU
print("Đang đọc dữ liệu...")
df = pd.read_csv(DATA_FILE)
features = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT']
X = df[features]

# TẠO NHÃN THẬT (GROUND TRUTH)
# Dùng chính công thức chuẩn để tạo nhãn từ cột PM2.5 gốc
print("Đang tính toán nhãn thực tế từ số liệu trạm...")
y_true = df['pm25'].apply(pm25_to_label_final).values

# 4. ĐÁNH GIÁ MODEL
models_to_test = ['rf_reg.pkl', 'xgb_reg.pkl']

print(f"\n{'='*10} BẮT ĐẦU ĐÁNH GIÁ (DỰA TRÊN CÔNG THỨC CHUẨN) {'='*10}")

for model_name in models_to_test:
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file: {model_name}")
        continue
        
    print(f"\nĐang đánh giá: {model_name}...")
    model = joblib.load(model_path)
    
    # A. Dự báo ra nồng độ PM2.5 (Số thực)
    y_pred_pm25 = model.predict(X)
    
    # B. Tính AQI từ PM2.5 dự báo -> Quy ra nhãn
    # Lưu ý: Phải dùng cùng một hàm pm25_to_label_final để đảm bảo công bằng
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