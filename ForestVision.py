import datetime as dt
from typing import List
import streamlit as st
import numpy as np
import requests
from utils.map_injection import inject_map
from utils.html_injection import (
    inject_metric_bar,
    inject_bold_centered,
    inject_total_change
)
from utils.metrics_processing import (
    label,
    convert_to_ha,
    calculate_metrics,
)
from utils.response_parsing import (
    request_satellite_images,
    parse_response,
)
from utils.graphics_processing import (
    overlay_vector_on_image,
    bw_to_color
)

# ── CONSTANTS & CONFIG ──────────────────────────────────────
DARK_BLUE = "#262730"
LIGHT_RED = "#994636"
LIGHT_GREEN = "#00B272"
LOGO_URL = 'frontend/images/GreenVision.png'
LEGEND_URL = 'https://vikfalk.github.io/deforestation_frontend/images/legend.png'
BRAZIL_ICON_URL = 'https://vikfalk.github.io/deforestation_frontend/images/brazil_example.png'
BOLIVIA_ICON_URL = 'https://vikfalk.github.io/deforestation_frontend/images/bolivia_example.png'
API_URL = "http://localhost:8080"

st.set_page_config(page_title="GreenVision", page_icon="🌳", layout="wide")

# ── SESSION STATE ───────────────────────────────────────────
st.session_state.setdefault('latitude_input', '-8.49')
st.session_state.setdefault('longitude_input', '-55.26')
st.session_state.setdefault('output', False)
st.session_state.setdefault('zoom', 6)
st.session_state.setdefault('start_timeframe', dt.date(2017, 6, 30))
st.session_state.setdefault('end_timeframe', dt.date(2024, 12, 31))

# ── HELPER FUNCTIONS ────────────────────────────────────────
def process_calculation_request(latitude, longitude, start_date, end_date):
    try:
        with st.spinner('Fetching Satellite Data...'):
            st.session_state.response = request_satellite_images(
                url=API_URL, latitude=latitude, longitude=longitude,
                start_date=start_date.strftime("%Y-%m-%d"), 
                end_date=end_date.strftime("%Y-%m-%d")
            )
            image_dates, segmented_images, raw_images = parse_response(st.session_state.response)
            
            # Metrics Calculation
            st.session_state.date_start, st.session_state.date_end = image_dates[0], image_dates[-1]
            metrics_df = calculate_metrics(image_dates, segmented_images).reset_index()
            st.session_state.cover_start = metrics_df.at[0, "coverage"]
            st.session_state.cover_end = metrics_df.at[1, "coverage"]
            st.session_state.cover_diff_rel = (st.session_state.cover_end - st.session_state.cover_start)
            st.session_state.cover_start_ha = convert_to_ha(st.session_state.cover_start)
            st.session_state.cover_end_ha = convert_to_ha(st.session_state.cover_end)
            st.session_state.cover_diff_ha = convert_to_ha(st.session_state.cover_diff_rel)

            # Image Processing (Forcing RGB)
            st.session_state.start_sat = np.array(raw_images[0])[:,:,:3]
            st.session_state.end_sat = np.array(raw_images[-1])[:,:,:3]
            st.session_state.start_overlay = overlay_vector_on_image(segmented_images[0], st.session_state.start_sat, LIGHT_GREEN)
            st.session_state.end_overlay = overlay_vector_on_image(segmented_images[-1], st.session_state.end_sat, LIGHT_GREEN)
            st.session_state.total_change_overlay = overlay_vector_on_image(segmented_images[0] - segmented_images[-1], st.session_state.end_overlay, LIGHT_RED)
            
            st.session_state.output = True
    except Exception as e:
        st.error(f"Processing Error: {e}")

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image(LOGO_URL, use_column_width=True)
    st.title('Location Selection')
    st.session_state.latitude_input = st.text_input('Latitude', st.session_state.latitude_input)
    st.session_state.longitude_input = st.text_input('Longitude', st.session_state.longitude_input)
    st.session_state.start_timeframe = st.date_input('Start Date', st.session_state.start_timeframe)
    st.session_state.end_timeframe = st.date_input('End Date', st.session_state.end_timeframe)
    
    if st.button('Calculate forest loss', use_container_width=True, type='primary'):
        process_calculation_request(st.session_state.latitude_input, st.session_state.longitude_input, 
                                    st.session_state.start_timeframe, st.session_state.end_timeframe)

# ── MAIN MAP ────────────────────────────────────────────────
inject_map(float(st.session_state.longitude_input), float(st.session_state.latitude_input), st.session_state.zoom)

# ── DYNAMIC OUTPUT PANEL ────────────────────────────────────
if st.session_state.output:
    st.write("---")
    inject_bold_centered("Output", font_em=2.5)
    
    # Summary Box
    st.info(f"Summary: Closest cloud-free images found between {st.session_state.date_start} and {st.session_state.date_end}.")

    # Main Grid Layout (Targeting your shared screenshot style)
    col_left, col_mid, col_right = st.columns([1, 1.2, 0.8])

    with col_left:
        with st.expander("Timeline Visuals", expanded=True):
            st.markdown("### Start Date")
            c1, c2 = st.columns(2)
            c1.image(st.session_state.start_sat, caption="Satellite", use_column_width=True)
            c2.image(st.session_state.start_overlay, caption="Segments", use_column_width=True)
            
            st.markdown("### End Date")
            c3, c4 = st.columns(2)
            c3.image(st.session_state.end_sat, caption="Satellite", use_column_width=True)
            c4.image(st.session_state.end_overlay, caption="Segments", use_column_width=True)

    with col_mid:
        with st.container():
            st.markdown("### Total Forest Change")
            st.image(st.session_state.total_change_overlay, use_column_width=True)
            st.image(LEGEND_URL, width=250)

    with col_right:
        st.markdown("### Metrics")
        tab1, tab2 = st.tabs(["Change in Percent", "Change in Hectar"])
        
        with tab1:
            inject_metric_bar("Start Cover", st.session_state.cover_start, "%")
            inject_metric_bar("End Cover", st.session_state.cover_end, "%")
            inject_total_change(st.session_state.cover_diff_rel, "%")
            
        with tab2:
            inject_metric_bar("Start Cover", st.session_state.cover_start_ha, " ha")
            inject_metric_bar("End Cover", st.session_state.cover_end_ha, " ha")
            inject_total_change(st.session_state.cover_diff_ha, " ha")

    # Lower Section: Deforestation Over Time
    st.write("---")
    inject_bold_centered("Deforestation Over Time", font_em=2)
    t_col1, t_col2 = st.columns(2)
    t_col1.image(st.session_state.start_sat, caption=st.session_state.date_start, use_column_width=True)
    t_col2.image(st.session_state.end_sat, caption=st.session_state.date_end, use_column_width=True)