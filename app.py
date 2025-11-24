import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import joblib
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- 1. CẤU HÌNH ---
DATA_ROOT = 'data_raw'
MODEL_DIR = 'models'

st.set_page_config(page_title="AQI Visualizer", layout="wide")
st.title("Visualizer Dự Báo AQI")

# --- 2. ĐỊNH NGHĨA MODEL ---
class AQIClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(AQIClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        out = self.dropout1(self.relu1(self.fc1(x)))
        out = self.dropout2(self.relu2(self.fc2(out)))
        out = self.dropout3(self.relu3(self.fc3(out)))
        out = self.fc4(out)
        return out

# --- 3. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        
        input_size = scaler.n_features_in_
        output_size = len(label_encoder.classes_)
        
        model = AQIClassifier(input_size, 128, 128, 128, output_size)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'aqi_classifier.pth'), map_location=torch.device('cpu')))
        model.eval()
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Lỗi load model: {e}"); return None, None, None

model, scaler, label_encoder = load_artifacts()

# --- 4. CẤU HÌNH MÀU SẮC CHUẨN AQI ---
AQI_COLORS = {
    "Tốt": "#00E400",       # Xanh lá
    "Trung bình": "#FFFF00",# Vàng
    "Kém": "#FF7E00",       # Cam
    "Xấu": "#FF0000",       # Đỏ
    "Rất xấu": "#8F3F97",   # Tím
    "Nguy hại": "#7E0023"   # Nâu đỏ
}

# --- 5. GIAO DIỆN ---
st.sidebar.header("Tùy Chọn")
input_date = st.sidebar.text_input("Nhập ngày (YYYYMMDD):", "20200101")

if st.sidebar.button("Chạy Dự Báo"):
    if not model: st.stop()
    
    # Tạo đường dẫn
    year, month, day = input_date[:4], input_date[4:6], input_date[6:8]
    day_folder = os.path.join(DATA_ROOT, year, month, day)
    dem_path = os.path.join(DATA_ROOT, "SQRT_SEA_DEM_LAT.tif")
    
    if not os.path.exists(day_folder) or not os.path.exists(dem_path):
        st.error(f"Không tìm thấy dữ liệu ngày {input_date} hoặc file địa hình.")
    else:
        with st.spinner("Đang tính toán..."):
            try:
                # 1. Đọc dữ liệu
                layers = []
                meteo_vars = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']
                for var in meteo_vars:
                    f_name = [f for f in os.listdir(day_folder) if f.startswith(var) and f.endswith('.tif')]
                    if not f_name: st.error(f"Thiếu {var}"); st.stop()
                    with rasterio.open(os.path.join(day_folder, f_name[0])) as src:
                        layers.append(np.nan_to_num(src.read(1), nan=0.0))
                
                with rasterio.open(dem_path) as src:
                    layers.append(np.nan_to_num(src.read(1), nan=0.0))
                
                # 2. Xử lý & Dự báo
                stack = np.dstack(layers)
                rows, cols, _ = stack.shape
                X_flat = stack.reshape(-1, 6)
                
                # Scale & Predict
                # Tạo tên cột giả để scaler không báo lỗi
                cols_name = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT']
                X_scaled = scaler.transform(pd.DataFrame(X_flat, columns=cols_name))
                
                with torch.no_grad():
                    outputs = model(torch.FloatTensor(X_scaled))
                    _, predicted = torch.max(outputs, 1)
                
                aqi_map = predicted.numpy().reshape(rows, cols)
                
                # 3. TẠO BẢNG MÀU ĐỘNG (DYNAMIC COLORMAP)
                classes = label_encoder.classes_ 
                
                # Tạo list màu tương ứng với từng lớp
                map_colors = []
                for c in classes:
                    # Lấy màu từ từ điển, nếu không có thì để màu xám
                    color = AQI_COLORS.get(str(c).strip(), "#808080")
                    map_colors.append(color)
                
                cmap = ListedColormap(map_colors)
                bounds = np.arange(len(classes) + 1) - 0.5
                norm = BoundaryNorm(bounds, cmap.N)

                # 4. HIỂN THỊ KẾT QUẢ
                st.success("Dự báo hoàn tất!")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    # Vẽ bản đồ với bảng màu đã chuẩn hóa
                    im = ax.imshow(aqi_map, cmap=cmap, norm=norm)
                    ax.axis('off')
                    ax.set_title(f"Bản đồ phân vùng AQI ngày {input_date}")
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Chú thích")
                    for i, c in enumerate(classes):
                        color = map_colors[i]
                        # Hiển thị ô màu + Tên lớp
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="width: 20px; height: 20px; background-color: {color}; margin-right: 10px; border: 1px solid #ccc;"></div>
                                <span style="font-size: 16px;">{c}</span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"Lỗi: {e}")