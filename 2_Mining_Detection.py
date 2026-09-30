"""
pages/2_Mining_Detection.py
Illegal Mining Detection using U-Net CNN + GEE Sentinel-2
"""

import datetime as dt
import streamlit as st
import numpy as np
import requests
import io
import base64
from PIL import Image, ImageDraw, ImageColor
import pydeck as pdk

# ── Config ────────────────────────────────────────────────────
LOCAL_URL  = "http://localhost:8080"
API_URL    = LOCAL_URL

MINING_RED    = "#FF4500"
SAFE_GREEN    = "#00B272"
WARN_ORANGE   = "#FFA500"

KNOWN_MINING_SITES = [
    {
        "name": "Jharia Coal Mine — Jharkhand, India",
        "lat": 23.75, "lon": 86.42,
        "date": "2023-01-01",
        "desc": "One of India's largest coal mining areas"
    },
    {
        "name": "Singrauli — Madhya Pradesh, India",
        "lat": 24.20, "lon": 82.67,
        "date": "2023-01-01",
        "desc": "Major open-cast coal mining region"
    },
    {
        "name": "Madre de Dios — Peru (Illegal Gold)",
        "lat": -12.50, "lon": -70.00,
        "date": "2023-01-01",
        "desc": "Illegal gold mining destroying Amazon rainforest"
    },
    {
        "name": "Amazon Illegal Mining — Brazil",
        "lat": -6.00, "lon": -55.00,
        "date": "2023-01-01",
        "desc": "Illegal mining hotspot in Brazilian Amazon"
    },
]

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Mining Detection — GreenVision",
    page_icon="⛏️",
    layout="wide"
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 25% !important; }
    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value { font-size: 2.5em; font-weight: bold; }
    .metric-label { font-size: 0.9em; color: #888; margin-top: 5px; }
    .alert-high   { color: #FF4500; }
    .alert-medium { color: #FFA500; }
    .alert-low    { color: #00B272; }
    .site-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────

def base64_to_numpy(img_b64):
    img_bytes = io.BytesIO(base64.b64decode(img_b64))
    return np.load(img_bytes)


def call_mining_api(lat, lon, date):
    """Call backend /detect_mining endpoint."""
    if hasattr(date, 'strftime'):
        date = date.strftime("%Y-%m-%d")
    resp = requests.get(
        f"{API_URL}/detect_mining",
        params={"latitude": lat, "longitude": lon, "date": date},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()


def create_mining_overlay(rgb_array, mining_mask):
    base = Image.fromarray(rgb_array, mode='RGB').convert('RGBA')
    red_layer = np.zeros((512, 512, 4), dtype=np.uint8)
    red_layer[:, :, 0] = 255
    red_layer[:, :, 1] = 50
    red_layer[:, :, 2] = 0
    # Strong alpha where mining detected
    red_layer[:, :, 3] = (mining_mask > 0.1).astype(np.uint8) * 180
    overlay = Image.fromarray(red_layer, mode='RGBA')
    result = Image.alpha_composite(base, overlay)
    return result.convert('RGB')

def create_heatmap_image(prob_map):
    # Normalize properly
    prob_min = prob_map.min()
    prob_max = prob_map.max()
    if prob_max - prob_min > 0:
        prob_norm = (prob_map - prob_min) / (prob_max - prob_min)
    else:
        prob_norm = prob_map
    
    prob_uint8 = (prob_norm * 255).astype(np.uint8)
    heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    heatmap[:, :, 0] = prob_uint8                              # Red channel
    heatmap[:, :, 1] = (prob_uint8 * 0.3).astype(np.uint8)   # Green channel
    heatmap[:, :, 2] = np.zeros_like(prob_uint8)              # Blue = 0
    return Image.fromarray(heatmap)


def get_severity(mining_pct):
    if mining_pct > 15:
        return "🔴 HIGH", "alert-high"
    elif mining_pct > 5:
        return "🟠 MEDIUM", "alert-medium"
    else:
        return "🟢 LOW", "alert-low"


def draw_map(lat, lon, zoom=10):
    view_state = pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=zoom)
    polygon_data = [{
        "polygon": [
            [float(lon)-0.023, float(lat)-0.023],
            [float(lon)+0.023, float(lat)-0.023],
            [float(lon)+0.023, float(lat)+0.023],
            [float(lon)-0.023, float(lat)+0.023],
            [float(lon)-0.023, float(lat)-0.023],
        ]
    }]
    layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        get_polygon="polygon",
        get_fill_color=[255, 69, 0, 60],
        get_line_color=[255, 69, 0, 200],
        get_line_width=3,
    )
    deck = pdk.Deck(
        initial_view_state=view_state,
        layers=[layer],
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )
    st.pydeck_chart(deck, use_container_width=True)


# ── Session State Init ────────────────────────────────────────
st.session_state.setdefault('mining_result', None)
st.session_state.setdefault('mining_lat', 23.75)
st.session_state.setdefault('mining_lon', 86.42)
st.session_state.setdefault('mining_date', dt.date(2023, 1, 1))


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛏️ Mining Detection")
    st.markdown("Detect illegal mining using **U-Net CNN** on Sentinel-2 imagery")
    st.markdown("---")

    st.markdown("### 📍 Custom Location")
    lat  = st.number_input("Latitude",  value=float(st.session_state.mining_lat),  format="%.4f")
    lon  = st.number_input("Longitude", value=float(st.session_state.mining_lon), format="%.4f")
    date = st.date_input(
        "Date", value=st.session_state.mining_date,
        min_value=dt.date(2017, 1, 1),
        max_value=dt.date(2025, 12, 31)
    )

    detect_btn = st.button("🔍 Detect Mining Activity", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌍 Known Mining Hotspots")
    for site in KNOWN_MINING_SITES:
        if st.button(f"📌 {site['name']}", use_container_width=True, key=site['name']):
            st.session_state.mining_lat  = site['lat']
            st.session_state.mining_lon  = site['lon']
            st.session_state.mining_date = dt.date.fromisoformat(site['date'])
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown("""
    **Architecture:** U-Net CNN  
    **Framework:** PyTorch  
    **Input:** 6-band spectral features  
    **Features:** NDVI, BSI, NDWI, Brightness, Red/NIR ratio, Green/Blue ratio  
    **Training:** Weakly supervised on Sentinel-2  
    """)


# ── Main Content ──────────────────────────────────────────────
st.markdown("# ⛏️ Illegal Mining Detection")
st.markdown("**AI-powered detection of illegal mining activity using U-Net deep learning on Sentinel-2 satellite imagery**")
st.markdown("---")

# Map always visible
col_map, col_info = st.columns([2, 1])
with col_map:
    st.markdown("### 🗺️ Analysis Region")
    draw_map(lat, lon)

with col_info:
    st.markdown("### 📊 Location Details")
    st.markdown(f"""
    | Field | Value |
    |---|---|
    | **Latitude** | `{lat}` |
    | **Longitude** | `{lon}` |
    | **Date** | `{date}` |
    | **Data Source** | Sentinel-2 L2A |
    | **Model** | U-Net CNN |
    | **Resolution** | 10m/pixel |
    """)
    st.markdown("---")
    st.info("👈 Select a known hotspot from sidebar or enter custom coordinates, then click **Detect Mining Activity**")


# ── Detection ─────────────────────────────────────────────────
if detect_btn:
    st.session_state.mining_lat  = lat
    st.session_state.mining_lon  = lon
    st.session_state.mining_date = date
    st.session_state.mining_result = None

    with st.spinner("🛰️ Fetching satellite image from GEE..."):
        try:
            result = call_mining_api(lat, lon, date)
            if result.get("error"):
                st.error(f"❌ {result['error']}")
            else:
                st.session_state.mining_result = result
                st.success(f"✅ Image fetched: {result['actual_date']}")
        except Exception as e:
            st.error(f"❌ API Error: {e}")


# ── Results ───────────────────────────────────────────────────
if st.session_state.mining_result:
    result = st.session_state.mining_result
    mining_pct = result.get("mining_percentage", 0)
    severity, sev_class = get_severity(mining_pct)

    st.markdown("---")
    st.markdown("## Detection Results")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(" Image Date",     result.get("actual_date", "N/A"))
    with m2:
        st.metric(" Mining Coverage", f"{mining_pct:.1f}%")
    with m3:
        st.metric(" Severity",        severity)
    with m4:
        clean_pct = 100 - mining_pct
        st.metric(" Clean Area",      f"{clean_pct:.1f}%")

    st.markdown("---")

    # Images
    orig_arr    = base64_to_numpy(result["original_img"])
    mask_arr    = base64_to_numpy(result["mining_mask"])
    prob_arr    = base64_to_numpy(result["prob_map"])
    st.write(f"Debug: Mining pixels = {int(mask_arr.sum())}, Max value = {mask_arr.max():.3f}")
    overlay_img = create_mining_overlay(orig_arr, mask_arr)
    heatmap_img = create_heatmap_image(prob_arr)

    st.markdown("### 🛰️ Satellite Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(orig_arr, caption=f"📡 Original Satellite — {result['actual_date']}",
                 use_column_width=True)
    with c2:
        st.image(overlay_img, caption="⛏️ Mining Detection Overlay (Red = Mining)",
                 use_column_width=True)
    with c3:
        st.image(heatmap_img, caption="🔥 U-Net Probability Heatmap",
                 use_column_width=True)

    # Alert box
    st.markdown("---")
    if mining_pct > 15:
        st.error(f"""
        ### 🚨 HIGH ALERT — Significant Mining Activity Detected!
        **{mining_pct:.1f}%** of the analyzed area shows signs of illegal mining activity.
        Coordinates: **{lat}, {lon}** | Date: **{result['actual_date']}**
        """)
    elif mining_pct > 5:
        st.warning(f"""
        ### ⚠️ MEDIUM ALERT — Moderate Mining Activity Detected
        **{mining_pct:.1f}%** of the analyzed area shows potential mining activity.
        Coordinates: **{lat}, {lon}** | Date: **{result['actual_date']}**
        """)
    else:
        st.success(f"""
        ### ✅ LOW RISK — Minimal Mining Activity
        Only **{mining_pct:.1f}%** of the area shows mining signatures.
        Coordinates: **{lat}, {lon}** | Date: **{result['actual_date']}**
        """)

    # Technical details expander
    with st.expander("🔬 Technical Details — U-Net Model Output"):
        st.markdown(f"""
        **Model:** U-Net CNN (PyTorch)
        **Input Features (6 channels):**
        - NDVI — Normalized Difference Vegetation Index
        - BSI — Bare Soil Index
        - NDWI — Normalized Difference Water Index
        - Brightness — Mean spectral brightness
        - Red/NIR ratio — Disturbed land signature
        - Green/Blue ratio — Sediment detection

        **Output:** Per-pixel mining probability map (0-1)
        **Threshold:** 0.4 (pixels above = mining detected)
        **Mining pixels detected:** {int(mask_arr.sum())} / {512*512} total
        **Mining coverage:** {mining_pct:.2f}%
        """)