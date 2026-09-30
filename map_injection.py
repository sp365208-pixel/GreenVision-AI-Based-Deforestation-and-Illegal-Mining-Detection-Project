import pydeck as pdk
import streamlit as st

# Constants for square area visualization
SQUARE_SIDE_LENGTH = 5.12  # km
EQUATOR_LATITUDINAL_DEGREE_IN_KM = 110.6
EQUATOR_LONGITUDINAL_DEGREE_IN_KM = 111.3
MEAN_GCS_DEGREE = (EQUATOR_LATITUDINAL_DEGREE_IN_KM + EQUATOR_LONGITUDINAL_DEGREE_IN_KM) / 2


def create_polygon_layer(lng, lat):
    """Creates a polygon (square) to highlight the deforestation area."""
    half_side_length = SQUARE_SIDE_LENGTH / 2 / MEAN_GCS_DEGREE
    lng, lat = float(lng), float(lat)

    # Define the square coordinates
    square_coords = [
        [lng - half_side_length, lat - half_side_length],
        [lng + half_side_length, lat - half_side_length],
        [lng + half_side_length, lat + half_side_length],
        [lng - half_side_length, lat + half_side_length],
        [lng - half_side_length, lat - half_side_length]
    ]

    polygon_data = [{
        "polygon": square_coords,
        "name": "Deforestation Area"
    }]

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        get_polygon="polygon",
        get_fill_color=[255, 0, 0, 60],  # Semi-transparent red
        get_line_color=[255, 255, 255],
        get_line_width=3,
        pickable=True,
        extruded=False,
    )
    return polygon_layer


def inject_map(lng, lat, zoom):
    """Injects an interactive map (no API key required)."""
    view_state = pdk.ViewState(
        longitude=float(lng),
        latitude=float(lat),
        zoom=zoom,
    )

    # 💡 Choose one of these:
    map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"  # Dark theme
    # map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"  # Light theme (nice contrast)

    deck = pdk.Deck(
        initial_view_state=view_state,
        layers=[create_polygon_layer(lng, lat)],
        map_provider=None,
        map_style=map_style,
        tooltip={"text": "Deforestation Area"},
    )

    st.pydeck_chart(deck, use_container_width=True)
