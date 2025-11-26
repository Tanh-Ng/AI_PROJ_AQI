import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import xgboost

# --- 1. CẤU HÌNH ---
DATA_ROOT = 'data_raw'
MODEL_DIR = 'models'

st.set_page_config(page_title="Pro AQI Visualizer", layout="wide")
st.title("Visualizer AQI")

# --- 2. HÀM HỖ TRỢ ---
def scan_data_structure(root_dir):
    structure = {}
    if not os.path.exists(root_dir): return structure
    for year in sorted(os.listdir(root_dir)):
        year_path = os.path.join(root_dir, year)
        if not os.path.isdir(year_path): continue
        year_data = {}
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path): continue
            valid_days = [d for d in sorted(os.listdir(month_path)) 
                          if any(f.endswith('.tif') for f in os.listdir(os.path.join(month_path, d)))]
            if valid_days: year_data[month] = valid_days
        if year_data: structure[year] = year_data
    return structure

# Dải màu Gradient nhiệt (Jet style) để nhìn rõ biến động
def get_heatmap_cmap():
    # Xanh (Sạch) -> Vàng -> Cam -> Đỏ -> Tím -> Đen (Bẩn)
    colors = ["#00008B", "#00BFFF", "#00FF00", "#FFFF00", "#FF7F50", "#FF0000", "#8B0000"]
    return LinearSegmentedColormap.from_list("custom_heat", colors, N=256)

# --- 3. GIAO DIỆN CẤU HÌNH ---
st.sidebar.header("Cấu Hình")

# A. Chọn Model
ml_models = [f for f in os.listdir(MODEL_DIR) if '_reg.pkl' in f or '_log.pkl' in f]
if not ml_models:
    st.error("Chưa có model Hồi quy! Chạy 'python train_reg.py'.")
    st.stop()

selected_model_file = st.sidebar.selectbox("Chọn Model:", ml_models, index=len(ml_models)-1)

# B. Chọn Ngày
data_tree = scan_data_structure(DATA_ROOT)
if not data_tree: st.error("Không tìm thấy data!"); st.stop()

years = list(data_tree.keys())
sel_year = st.sidebar.selectbox("Năm", years)
months = list(data_tree[sel_year].keys())
sel_month = st.sidebar.selectbox("Tháng", months)
days = data_tree[sel_year][sel_month]
sel_day = st.sidebar.selectbox("Ngày", days)

st.sidebar.markdown("---")
# C. TÙY CHỌN HIỂN THỊ (TÍNH NĂNG MỚI)
st.sidebar.subheader("Hiển Thị")
use_auto_contrast = st.sidebar.checkbox("Tăng cường độ tương phản", value=True, help="Tự động co giãn màu sắc để làm rõ các biến động nhỏ nhất.")
show_values = st.sidebar.checkbox("Hiển thị chú thích số liệu", value=True)

# --- 4. XỬ LÝ CHÍNH ---
if st.sidebar.button("Chạy Dự Báo"):
    model_path = os.path.join(MODEL_DIR, selected_model_file)
    with st.spinner(f"Đang tải {selected_model_file}..."):
        model = joblib.load(model_path)

    day_folder = os.path.join(DATA_ROOT, sel_year, sel_month, sel_day)
    dem_path = os.path.join(DATA_ROOT, "SQRT_SEA_DEM_LAT.tif")
    
    if not os.path.exists(day_folder): st.error("Lỗi đường dẫn data"); st.stop()

    with st.spinner("Đang tính toán heatmap..."):
        try:
            layers = []
            # 1. Đọc DEM
            with rasterio.open(dem_path) as src:
                dem_raw = src.read(1)
                mask = (dem_raw == src.nodata) | np.isnan(dem_raw)
                layers.append(np.nan_to_num(dem_raw, nan=0.0))
            
            # 2. Đọc Khí tượng
            for var in ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']:
                f = [x for x in os.listdir(day_folder) if x.startswith(var)][0]
                with rasterio.open(os.path.join(day_folder, f)) as src:
                    data = src.read(1)
                    mask = mask | (data == src.nodata) | np.isnan(data)
                    layers.append(np.nan_to_num(data, nan=0.0))
            
            # 3. Predict
            stack = np.dstack(layers[1:] + [layers[0]])
            rows, cols, _ = stack.shape
            X_df = pd.DataFrame(stack.reshape(-1, 6), columns=['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT'])
            
            pred_raw = model.predict(X_df)
            
            # Xử lý nếu là model Log
            if "_log" in selected_model_file:
                pred_real = np.expm1(pred_raw)
            else:
                pred_real = pred_raw
            
            # 4. Xử lý Hiển thị (Auto Contrast Logic)
            pm25_map = pred_real.reshape(rows, cols)
            map_masked = np.ma.masked_where(mask, pm25_map)
            
            # Tính toán min/max cho màu sắc
            if use_auto_contrast:
                valid_data = pm25_map[~mask]
                vmin = np.percentile(valid_data, 2)
                vmax = np.percentile(valid_data, 98)
            else:
                # Dùng chuẩn cố định (0 - 300)
                vmin = 0
                vmax = 300
            
            st.success(f"Kết quả ({selected_model_file})")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 8))
                cmap = get_heatmap_cmap()
                cmap.set_bad(color='white', alpha=0)
                
                im = ax.imshow(map_masked, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')
                ax.set_title(f"Bản đồ nồng độ PM2.5 ngày {sel_day}/{sel_month}/{sel_year}", fontsize=16)
                
                if show_values:
                    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                    cbar.set_label('PM2.5 (µg/m³)')
                
                st.pyplot(fig)
            
            with col2:
                st.info("**Thống kê ngày:**")
                st.metric("Thấp nhất", f"{np.min(pred_real):.1f}")
                st.metric("Cao nhất", f"{np.max(pred_real):.1f}")
                st.metric("Trung bình", f"{np.mean(pred_real):.1f}")
                
                st.write("---")
        except Exception as e: st.error(f"Lỗi: {e}")