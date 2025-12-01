import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import xgboost

# --- 1. CẤU HÌNH ---
DATA_ROOT = 'data_raw'
MODEL_DIR = 'models'

st.set_page_config(page_title="AQI Visualizer VN", layout="wide")
st.title("Chất Lượng Không Khí Hà Nội")

# --- 2. CÔNG THỨC TÍNH AQI ---
def calculate_aqi_scalar(pm25):
    """
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

# Vectorize để chạy nhanh trên ma trận
v_calculate_aqi = np.vectorize(calculate_aqi_scalar)

# --- 3. HÀM HỖ TRỢ ---
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

# --- 4. GIAO DIỆN ---
st.sidebar.header("Cấu Hình")

ml_models = [f for f in os.listdir(MODEL_DIR) if '_reg.pkl' in f]
if not ml_models: st.error("Chưa có model Regression!"); st.stop()
selected_model_file = st.sidebar.selectbox("Chọn Model:", ml_models)

data_tree = scan_data_structure(DATA_ROOT)
if not data_tree: st.error("Không tìm thấy data!"); st.stop()
years = list(data_tree.keys())
sel_year = st.sidebar.selectbox("Năm", years)
months = list(data_tree[sel_year].keys())
sel_month = st.sidebar.selectbox("Tháng", months)
days = data_tree[sel_year][sel_month]
sel_day = st.sidebar.selectbox("Ngày", days)

view_mode = st.sidebar.radio("Hiển thị:", ["Chỉ số AQI", "Nồng độ PM2.5"])

# --- 5. XỬ LÝ CHÍNH ---
if st.sidebar.button("Chạy Dự Báo"):
    model_path = os.path.join(MODEL_DIR, selected_model_file)
    with st.spinner(f"Đang chạy mô hình..."):
        model = joblib.load(model_path)

    day_folder = os.path.join(DATA_ROOT, sel_year, sel_month, sel_day)
    dem_path = os.path.join(DATA_ROOT, "SQRT_SEA_DEM_LAT.tif")
    
    with st.spinner("Đang tính toán..."):
        try:
            layers = []
            with rasterio.open(dem_path) as src:
                dem_raw = src.read(1)
                mask = (dem_raw == src.nodata) | np.isnan(dem_raw)
                layers.append(np.nan_to_num(dem_raw, nan=0.0))
            
            for var in ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']:
                f = [x for x in os.listdir(day_folder) if x.startswith(var)][0]
                with rasterio.open(os.path.join(day_folder, f)) as src:
                    layers.append(np.nan_to_num(src.read(1), nan=0.0))
            
            # Predict
            stack = np.dstack(layers[1:] + [layers[0]])
            rows, cols, _ = stack.shape
            pm25_pred = model.predict(stack.reshape(-1, 6)).reshape(rows, cols)
            
            # Tính AQI chuẩn xác
            aqi_pred = v_calculate_aqi(pm25_pred)
            
            st.success(f"Kết quả ngày {sel_day}/{sel_month}/{sel_year}")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if "AQI" in view_mode:
                    data_to_plot = aqi_pred
                    title = "Chỉ số AQI (0-500)"
                    
                    # BẢNG MÀU 5 CẤP (Gộp >200 thành Tím)
                    colors = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97"]
                    cmap = ListedColormap(colors)
                    
                    # Bounds: Tô màu Tím cho tất cả giá trị từ 200 đến 500
                    bounds = [0, 50, 100, 150, 200, 500] 
                    norm = BoundaryNorm(bounds, cmap.N)
                else:
                    data_to_plot = pm25_pred
                    title = "Nồng độ Bụi PM2.5 (µg/m³)"
                    cmap = plt.get_cmap("jet")
                    norm = None

                map_masked = np.ma.masked_where(mask, data_to_plot)
                cmap.set_bad('white', 0)
                
                im = ax.imshow(map_masked, cmap=cmap, norm=norm)
                ax.axis('off'); ax.set_title(title)
                
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                if "AQI" in view_mode:
                    cbar.set_ticks([25, 75, 125, 175, 350])
                    cbar.set_ticklabels(["Tốt", "TB", "Kém", "Xấu", "Rất xấu"])
                
                st.pyplot(fig)
                
            with col2:
                if "AQI" in view_mode:
                    st.write("**Thang đo:**")
                    st.markdown("🟢 **0-50:** Tốt")
                    st.markdown("🟡 **51-100:** Trung bình")
                    st.markdown("🟠 **101-150:** Kém")
                    st.markdown("🔴 **151-200:** Xấu")
                    st.markdown("🟣 **>200:** Rất xấu")
                    
                    st.info(f"Max AQI: {np.max(aqi_pred):.1f}")
                    st.info(f"Min AQI: {np.min(aqi_pred):.1f}")
                else:
                    st.metric("Max PM2.5", f"{np.max(pm25_pred):.1f}")
                    st.metric("Min PM2.5", f"{np.min(pm25_pred):.1f}")

        except Exception as e: st.error(f"Lỗi: {e}")