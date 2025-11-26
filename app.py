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
st.title("Hệ thống Dự báo Chất lượng Không khí (AQI)")

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

# --- 3. CÁC HÀM HỖ TRỢ QUAN TRỌNG ---

def scan_data_structure(root_dir):
    """Quét cấu trúc thư mục Năm/Tháng/Ngày để tạo menu"""
    structure = {}
    if not os.path.exists(root_dir): return structure

    for year in sorted(os.listdir(root_dir)):
        year_path = os.path.join(root_dir, year)
        if not os.path.isdir(year_path): continue
        
        year_data = {}
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path): continue
            
            valid_days = []
            for day in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day)
                if not os.path.isdir(day_path): continue
                if any(f.endswith('.tif') for f in os.listdir(day_path)):
                    valid_days.append(day)
            
            if valid_days:
                year_data[month] = valid_days
        
        if year_data:
            structure[year] = year_data
            
    return structure

def get_aqi_color(label_name):
    name_clean = str(label_name).strip().lower()
    
    # Danh sách ưu tiên: (Từ khóa, Mã màu)
    priority_map = [
        ("nguy hại", "#7E0023"),  # Nâu (Check trước tiên)
        ("rất xấu", "#8F3F97"),   # Tím (Check trước Xấu)
        ("xấu", "#FF0000"),       # Đỏ
        ("kém", "#FF7E00"),       # Cam
        ("trung bình", "#FFFF00"),# Vàng
        ("tốt", "#00E400")        # Xanh
    ]
    
    for key, color in priority_map:
        if key in name_clean:
            return color
            
    return "#808080" # Màu xám nếu không tìm thấy

# --- 4. LOAD ARTIFACTS ---
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

# --- 5. GIAO DIỆN CHỌN NGÀY (MENU DROPDOWN) ---
st.sidebar.header("Chọn Thời Gian")

data_tree = scan_data_structure(DATA_ROOT)

if not data_tree:
    st.sidebar.error(f"Không tìm thấy dữ liệu trong '{DATA_ROOT}'")
    st.stop()

# Chọn Năm
years = list(data_tree.keys())
sel_year = st.sidebar.selectbox("Năm", years)

# Chọn Tháng
months = list(data_tree[sel_year].keys())
sel_month = st.sidebar.selectbox("Tháng", months)

# Chọn Ngày
days = data_tree[sel_year][sel_month]
sel_day = st.sidebar.selectbox("Ngày", days)

# Hiển thị thông báo
st.sidebar.success(f"Đã chọn: {sel_day}/{sel_month}/{sel_year}")

# --- 6. XỬ LÝ CHÍNH ---
if st.sidebar.button("Chạy Dự Báo"):
    if not model: st.stop()
    
    # Tạo đường dẫn từ menu đã chọn
    day_folder = os.path.join(DATA_ROOT, sel_year, sel_month, sel_day)
    dem_path = os.path.join(DATA_ROOT, "SQRT_SEA_DEM_LAT.tif")
    
    if not os.path.exists(day_folder) or not os.path.exists(dem_path):
        st.error(f"Lỗi đường dẫn dữ liệu: {day_folder}")
    else:
        with st.spinner("Đang xử lý dữ liệu vệ tinh..."):
            try:
                layers = []
                
                # 1. Đọc DEM trước để lấy Mask
                with rasterio.open(dem_path) as src:
                    dem_raw = src.read(1)
                    mask = (dem_raw == src.nodata) | np.isnan(dem_raw)
                    layers.append(np.nan_to_num(dem_raw, nan=0.0))

                # 2. Đọc khí tượng
                meteo_vars = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']
                for var in meteo_vars:
                    f_name = [f for f in os.listdir(day_folder) if f.startswith(var) and f.endswith('.tif')]
                    if not f_name: st.error(f"Thiếu file: {var}"); st.stop()
                    
                    with rasterio.open(os.path.join(day_folder, f_name[0])) as src:
                        data = src.read(1)
                        mask = mask | (data == src.nodata) | np.isnan(data)
                        layers.append(np.nan_to_num(data, nan=0.0))
                
                # 3. Sắp xếp layer: [PRES, RH, WSPD, TMP, TP, DEM]
                ordered_layers = layers[1:] + [layers[0]]

                # 4. Dự báo
                stack = np.dstack(ordered_layers)
                rows, cols, _ = stack.shape
                X_flat = stack.reshape(-1, 6)
                
                cols_name = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT']
                X_scaled = scaler.transform(pd.DataFrame(X_flat, columns=cols_name))
                
                with torch.no_grad():
                    outputs = model(torch.FloatTensor(X_scaled))
                    _, predicted = torch.max(outputs, 1)
                
                aqi_map = predicted.numpy().reshape(rows, cols)
                aqi_map_masked = np.ma.masked_where(mask, aqi_map)
                
                # 5. Xử lý màu sắc (CHUẨN HÓA)
                classes = label_encoder.classes_ 
                map_colors = []
                
                # Duyệt qua từng lớp và tìm màu đúng chuẩn
                for c in classes:
                    color = get_aqi_color(c)
                    map_colors.append(color)
                
                cmap = ListedColormap(map_colors)
                cmap.set_bad(color='white', alpha=0) 
                
                bounds = np.arange(len(classes) + 1) - 0.5
                norm = BoundaryNorm(bounds, cmap.N)

                # 6. Hiển thị Kết quả
                st.success("Dự báo hoàn tất")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(aqi_map_masked, cmap=cmap, norm=norm)
                    ax.axis('off')
                    ax.set_title(f"Bản đồ AQI - {sel_day}/{sel_month}/{sel_year}")
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Chú thích")
                    # Hiển thị bảng chú thích với màu chuẩn đã map
                    for i, c in enumerate(classes):
                        color = map_colors[i]
                        st.markdown(
                            f"""<div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <div style="width: 24px; height: 24px; background-color: {color}; margin-right: 12px; border: 1px solid #ccc; border-radius: 4px;"></div>
                                <span style="font-size: 16px;">{c}</span></div>""", 
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"Lỗi chi tiết: {e}")