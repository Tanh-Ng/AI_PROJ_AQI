import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# ==============================================================================
# 1. ĐỊNH NGHĨA LẠI CÁCH CHIA NHÓM AQI (Phải giống lúc Train)
# ==============================================================================
def get_aqi_group(pm25):
    # US EPA Breakpoints
    if pm25 <= 12.0: return 0      # Tốt (Good)
    elif pm25 <= 35.4: return 1    # Trung bình (Moderate)
    elif pm25 <= 55.4: return 2    # Kém (Unhealthy for Sensitive)
    elif pm25 <= 150.4: return 3   # Xấu (Unhealthy)
    else: return 4                 # Rất xấu/Nguy hại (Very Unhealthy/Hazardous)

# Tên các nhãn để hiển thị biểu đồ cho đẹp
LABEL_NAMES = ['Tốt', 'Trung bình', 'Kém', 'Xấu', 'Rất xấu']

# ==============================================================================
# 2. HÀM ĐÁNH GIÁ CHUYÊN SÂU
# ==============================================================================
def evaluate_saved_model(model_path, X_test, y_test, model_name):
    print(f"\n{'='*20} ĐANG ĐÁNH GIÁ: {model_name} {'='*20}")
    
    # Load model từ file
    try:
        model = joblib.load(model_path)
        print(f"-> Đã load model từ: {model_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {model_path}")
        return

    # 1. Dự báo (Regression)
    y_pred = model.predict(X_test)
    
    # 2. Chỉ số Regression
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n[Chỉ số Hồi quy]")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 3. Chuyển đổi sang bài toán Phân loại (Classification)
    y_test_class = [get_aqi_group(y) for y in y_test]
    y_pred_class = [get_aqi_group(y) for y in y_pred]

    # 4. Classification Report (Precision, Recall, F1)
    print(f"\n[Báo cáo Phân loại AQI]")
    print(classification_report(y_test_class, y_pred_class, 
                                target_names=LABEL_NAMES, zero_division=0))

    # 5. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test_class, y_pred_class)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f'Confusion Matrix - {model_name}\n(R2: {r2:.2f})')
    plt.xlabel('Dự báo (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. MAIN RUN
# ==============================================================================
if __name__ == "__main__":
    print("Đang load dữ liệu Test...")
    try:
        # Load dữ liệu Test đã lưu từ bước trước
        X_test = joblib.load('models/X_test.pkl')
        y_test = joblib.load('models/y_test.pkl')
        print(f"Load thành công X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Gọi hàm đánh giá cho từng model
        evaluate_saved_model('models/best_rf_model.pkl', X_test, y_test, "Random Forest")
        evaluate_saved_model('models/best_xgb_model.pkl', X_test, y_test, "XGBoost")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu test hoặc model trong thư mục 'models/'.")
        print("Hãy chắc chắn bạn đã chạy file Train và hàm save_artifacts() trước.")