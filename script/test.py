import streamlit as st
import folium
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64
import os

# === Bounding box ===
TARGET_DOMAIN_LAT_MIN, TARGET_DOMAIN_LAT_MAX = -41.989723, 27.232262
TARGET_DOMAIN_LON_MIN, TARGET_DOMAIN_LON_MAX = -27.161226, 79.549774
bounds = [[TARGET_DOMAIN_LAT_MIN, TARGET_DOMAIN_LON_MIN], [TARGET_DOMAIN_LAT_MAX, TARGET_DOMAIN_LON_MAX]]
map_center = [(TARGET_DOMAIN_LAT_MIN + TARGET_DOMAIN_LAT_MAX) / 2, (TARGET_DOMAIN_LON_MIN + TARGET_DOMAIN_LON_MAX) / 2 - 15]

def get_file(dt):
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    hour = dt.strftime("%H")
    minute = dt.strftime("%M")
    path_core = f'/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/{year}/{month}'
    return f'{path_core}/{year}{month}{day}{hour}{minute}.nc'

@st.cache_data
def load_data_cached(file_path):
    if not os.path.exists(file_path):
        return None, None
    try:
        data = Dataset(file_path, mode='r')
        tir = np.array(data['tir'][0, :, :])
        cores = np.array(data['cores'][0, :, :])
        data.close()
        return tir, cores
    except Exception:
        return None, None

@st.cache_data
def array_to_masked_colormap_img(data, mask, cmap_name="plasma", vmin=None, vmax=None, alpha=1.0):
    data = np.nan_to_num(data)
    flipped_data = np.flipud(data)
    flipped_mask = np.flipud(mask.astype(bool))

    if np.any(flipped_mask):
        vmin = np.min(flipped_data[flipped_mask]) if vmin is None else vmin
        vmax = np.max(flipped_data[flipped_mask]) if vmax is None else vmax
    else:
        vmin, vmax = 0, 1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(flipped_data))
    rgba[..., -1] = np.where(flipped_mask, alpha, 0.0)

    img = (rgba * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="RGBA")
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}", vmin, vmax

@st.cache_data
def render_colorbar(cmap_name, vmin, vmax, label):
    fig, ax = plt.subplots(figsize=(3, 1))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return buf.getvalue()

# === Streamlit UI ===
st.set_page_config(layout="wide")

st.markdown("""
<style>
.fixed-title {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: white;
    border-bottom: 1px solid #ddd;
    padding: 10px 16px;
    font-size: 30px;
    font-weight: 600;
    z-index: 999;
}
.spacer {
    height: 60px;
}
</style>

<div class="fixed-title">
    Cloud Top Temperature and Cores Viewer
</div>
<div class="spacer"></div>
""", unsafe_allow_html=True)


date_col, _ = st.columns([1, 1])

with date_col:
    col_y, col_m, col_d, col_h, col_min = st.columns(5)

    with col_y:
        year = st.selectbox("Year", list(range(2004, 2024)), index=0)
    with col_m:
        month = st.selectbox("Month", list(range(1, 13)), index=0)
    with col_d:
        day = st.selectbox("Day", list(range(1, 32)), index=0)
    with col_h:
        hour = st.selectbox("Hour", list(range(0, 24)), index=12)
    with col_min:
        minute = st.selectbox("Minute", [0, 15, 30, 45], index=0)

    try:
        dt_selected = datetime(year, month, day, hour, minute)
    except ValueError:
        st.error("Invalid date (e.g. Feb 30). Please adjust.")
        dt_selected = None


file_path = get_file(dt_selected)
tir, cores = load_data_cached(file_path)

m = folium.Map(
    location=map_center,
    zoom_start=100,
    tiles=None,
    max_bounds=True,
    min_zoom=10,
    max_zoom=11
)
m.fit_bounds(bounds)

folium.TileLayer('CartoDB positron', name='Light', show=True).add_to(m)
folium.TileLayer('CartoDB dark_matter', name='Dark', show=False).add_to(m)
folium.TileLayer('OpenStreetMap', name='OpenStreetMap', show=False).add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    name='Esri Topo', show=False, attr='Tiles © Esri'
).add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    name='Esri Satellite', show=False, attr='Tiles © Esri'
).add_to(m)

colorbars = []
if tir is not None and cores is not None:
    tir_img, vmin_tir, vmax_tir = array_to_masked_colormap_img(tir, mask=(tir < 0), cmap_name="plasma", alpha=1.0)
    cores_img, vmin_cores, vmax_cores = array_to_masked_colormap_img(cores, mask=(cores > 0), cmap_name="viridis", alpha=1.0)

    ctt_layer = folium.FeatureGroup(name="CTT < 0°C", show=True)
    cores_layer = folium.FeatureGroup(name="Cores > 0", show=True)

    ImageOverlay(image=tir_img, bounds=bounds, opacity=1.0).add_to(ctt_layer)
    ImageOverlay(image=cores_img, bounds=bounds, opacity=1.0).add_to(cores_layer)

    ctt_layer.add_to(m)
    cores_layer.add_to(m)

    colorbars.append(("Temperature (C)", "plasma", vmin_tir, vmax_tir))
    colorbars.append(("Wavelet power", "viridis", vmin_cores, vmax_cores))

folium.LayerControl(collapsed=True).add_to(m)

left_col, right_col = st.columns([3, 1])
with left_col:
    st.empty()
    st_data = st_folium(m, width=1000, height=600)

with right_col:
    if colorbars:
        st.markdown("## Legends")
        for label, cmap, vmin, vmax in colorbars:
            cb_png = render_colorbar(cmap, vmin, vmax, label)
            st.image(cb_png, width=210)

        st.markdown("---")
        st.markdown(
            """
            ### About this viewer

            This tool displays real-time satellite-derived storm features over Southern Africa using
            Meteosat data (CH9 wavelet outputs).  
            Cloud-top temperatures (CTT) and wavelet-extracted cores highlight deep convection and storm potential.

            - Temporal resolution: **15 minutes**  
            - Grid spacing: **~3 km** (native MSG resolution)

            Data source: EUMETSAT via JASMIN
            """,
            unsafe_allow_html=True
        )


# === Fixed Footer ===
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: black;
    color: white;
    text-align: center;
    padding: 8px 0;
    font-size: 0.85em;
    z-index: 100;
    border-top: 1px solid #d3d3d3;
}
</style>
<div class="footer">
    Mendrika Rakotomanga
</div>
""", unsafe_allow_html=True)

