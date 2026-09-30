"""
pages/3_Chatbot_Assistant.py
ForestVision AI Chatbot - Search location coordinates + Mining information
"""

import streamlit as st
import requests
import json

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant — GreenVision",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 25% !important; }
    .chat-user {
        background: #1a3a5c;
        border-radius: 15px 15px 5px 15px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        color: #e2f0ff;
    }
    .chat-bot {
        background: #1a2a1a;
        border-radius: 15px 15px 15px 5px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        color: #e2ffe2;
        border-left: 3px solid #00B272;
    }
    .coord-box {
        background: #0e1a0e;
        border: 1px solid #00B272;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


# ── Geocoding using OpenStreetMap Nominatim (Free) ────────────

def get_coordinates(place_name):
    """
    Get coordinates for any place using OpenStreetMap Nominatim API.
    Completely free - no API key needed.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place_name,
            "format": "json",
            "limit": 1,
            "addressdetails": 1
        }
        headers = {"User-Agent": "ForestVision/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        if data:
            result = data[0]
            lat = float(result["lat"])
            lon = float(result["lon"])
            display_name = result.get("display_name", place_name)
            return {
                "success": True,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "display_name": display_name,
                "place": place_name
            }
        else:
            return {"success": False, "error": f"Location '{place_name}' not found."}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Claude API for Mining Information ────────────────────────

def get_mining_info(place_name):
    """
    Get mining information for a location using Claude API.
    """
    try:
        prompt = f"""You are an environmental intelligence assistant specializing in mining activities.

For the location: {place_name}

Please provide the following information in a structured way:
1. Known mining activities in this region (legal and illegal)
2. Types of minerals being mined
3. Organizations or companies that have official mining authority/licenses
4. Any known illegal mining issues
5. Environmental impact of mining in this area

Keep the response concise, factual, and informative. Format with clear sections.
If you don't have specific information, mention general mining context for the region."""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        data = response.json()
        if "content" in data and data["content"]:
            return data["content"][0]["text"]
        else:
            return "Sorry, could not fetch mining information at this time."

    except Exception as e:
        return f"Error fetching mining information: {str(e)}"


# ── Session State ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_coords" not in st.session_state:
    st.session_state.last_coords = None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.markdown("Ask me anything about any location!")
    st.markdown("---")
    st.markdown("### 💡 Example Questions:")
    
    example_questions = [
        "Where is Jharia Coal Mine?",
        "Give coordinates of Amazon rainforest",
        "Mining activities in Jharkhand",
        "Coordinates of Singrauli MP",
        "Who does mining in Madre de Dios Peru?",
        "Mining authority in Odisha",
    ]
    
    for q in example_questions:
        if st.button(f"💬 {q}", use_container_width=True, key=q):
            st.session_state.chat_history.append({
                "role": "user",
                "content": q
            })
            st.experimental_rerun()
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_coords = None
        st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("### 🛠️ Features:")
    st.markdown("""
    - 📍 Get coordinates of any location
    - ⛏️ Mining info & authorities  
    - 🏛️ Legal mining licenses
    - ⚠️ Illegal mining alerts
    - 🌍 Works for any location globally
    """)


# ── Main Content ──────────────────────────────────────────────
st.markdown("# 🤖 GreenVision AI Assistant")
st.markdown("**Search any location for coordinates and mining information**")
st.markdown("---")

# Display chat history
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center; padding:40px; color:#888;'>
            <h3>👋 Hello! I am the GreenVision AI Assistant</h3>
            <p>Ask me about any location to get:</p>
            <p>📍 <b>Coordinates</b> — Latitude & Longitude</p>
            <p>⛏️ <b>Mining Info</b> — Who is mining and with what authority</p>
            <p>⚠️ <b>Illegal Mining</b> — Known illegal activities in the area</p>
            <br>
            <p style='color:#00B272;'>Try asking: "Where is Jharia Coal Mine?" or "Mining in Amazon Brazil"</p>
        </div>
        """, unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-user'>
                👤 <b>You:</b><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-bot'>
                🤖 <b>GreenVision Assistant:</b><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)

            # Show coordinate box if available
            # if "coords" in msg:
            #     coords = msg["coords"]
            #     st.markdown(f"""
            #     <div class='coord-box'>
            #         📍 <b>Coordinates Found:</b><br>
            #         Latitude: <b>{coords['lat']}</b><br>
            #         Longitude: <b>{coords['lon']}</b>
            #     </div>
            #     """, unsafe_allow_html=True)

            #     col1, col2 = st.columns(2)
            #     with col1:
            #         if st.button(f"🌲 Use for Deforestation Analysis", 
            #                     key=f"defor_{i}",
            #                     use_container_width=True):
            #             st.session_state["auto_lat"] = coords["lat"]
            #             st.session_state["auto_lon"] = coords["lon"]
            #             st.switch_page("ForestVision.py")
            #     with col2:
            #         if st.button(f"⛏️ Use for Mining Detection",
            #                     key=f"mine_{i}",
            #                     use_container_width=True):
            #             st.session_state["mining_lat"] = coords["lat"]
            #             st.session_state["mining_lon"] = coords["lon"]
            #             st.switch_page("pages/2_Mining_Detection.py")


# ── Process pending user messages ────────────────────────────
def process_last_message():
    """Process the last user message and generate bot response."""
    if not st.session_state.chat_history:
        return
    
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] != "user":
        return
    
    user_input = last_msg["content"].strip()
    
    # Detect intent
    coord_keywords = ["coordinates", "coordinate", "where is", "location of", 
                      "lat", "lon", "where", "find", "locate", "map"]
    mining_keywords = ["mining", "mine", "mineral", "authority", "license", 
                       "illegal", "who is", "company", "organization"]
    
    is_coord = any(kw in user_input.lower() for kw in coord_keywords)
    is_mining = any(kw in user_input.lower() for kw in mining_keywords)

    # Extract place name (simple approach)
    place_name = user_input
    for kw in ["coordinates of", "where is", "location of", "find", 
                "locate", "mining in", "mining at", "who does mining in",
                "give coordinates of", "mining activities in"]:
        place_name = place_name.lower().replace(kw, "").strip()

    bot_response = {"role": "assistant", "content": ""}

    # Get coordinates
    coord_result = get_coordinates(place_name)

    if coord_result["success"]:
        bot_response["coords"] = {
            "lat": coord_result["lat"],
            "lon": coord_result["lon"]
        }
        response_text = f"""I found the location **{coord_result['display_name']}**\n\n
📍 **Latitude:** {coord_result['lat']}  
📍 **Longitude:** {coord_result['lon']}\n
You can directly use these coordinates in the Deforestation or Mining Detection modules!"""

        # If mining related, also get mining info
        if is_mining or not is_coord:
            response_text += "\n\n---\n\n⛏️ **Mining Information:**\n\n"
            mining_info = get_mining_info(place_name)
            response_text += mining_info
        
        bot_response["content"] = response_text

    else:
        # Just get mining info if coordinates not found
        if is_mining:
            mining_info = get_mining_info(user_input)
            bot_response["content"] = f"⛏️ **Mining Information for {user_input}:**\n\n{mining_info}"
        else:
            bot_response["content"] = f"Sorry, I could not find the location '{place_name}'. Please try a more specific location name."

    st.session_state.chat_history.append(bot_response)
    st.experimental_rerun()


# Process if last message is from user
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    with st.spinner("🤖 Searching for information..."):
        process_last_message()


# ── Chat Input ────────────────────────────────────────────────
st.markdown("---")
col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        "Ask anything...",
        placeholder="e.g. Where is Jharia Coal Mine? OR Mining activities in Odisha",
        label_visibility="collapsed",
        key="chat_input"
    )

with col_btn:
    send_btn = st.button("Send 🚀", type="primary", use_container_width=True)

if send_btn and user_input.strip():
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input.strip()
    })
    st.experimental_rerun()