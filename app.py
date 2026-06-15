import base64
from pathlib import Path

import streamlit as st

from rag_chain import get_answer

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
ASSETS = Path(__file__).parent / "assets"
favicon = str(ASSETS / "favicon.png")

st.set_page_config(
    page_title="UChicago MS-ADS Assistant",
    page_icon=favicon,
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# UChicago brand styling (maroon + white)
# ─────────────────────────────────────────────────────────────────────────────
MAROON = "#800000"

st.markdown(
    f"""
    <style>
    .stApp {{ background: #FFFFFF; }}

    /* Header banner */
    .uc-header {{
        display: flex; align-items: center; gap: 14px;
        border-bottom: 4px solid {MAROON};
        padding: 6px 0 16px 0; margin-bottom: 8px;
    }}
    .uc-header img.logo {{ height: 50px; width: auto; flex: 0 0 auto; }}
    .uc-title {{
        font-family: Georgia, "Times New Roman", serif;
        color: {MAROON};
        font-size: 2.4rem; font-weight: 700; line-height: 1.15; margin: 0;
    }}
    .uc-sub {{
        color: #4a4a4a; font-size: 1.2rem; margin-top: 2px; line-height: 1.3;
    }}

    /* Chat bubbles */
    [data-testid="stChatMessage"] {{ background: transparent; }}
    /* User messages: maroon accent */
    [data-testid="stChatMessageContent"] {{ font-size: 1rem; }}

    /* Buttons / inputs */
    .stChatInput textarea {{ border-radius: 10px !important; }}
    [data-testid="stChatInput"] {{ border: 1.5px solid #e3d6d6; border-radius: 12px; }}

    /* Sources expander */
    .streamlit-expanderHeader {{ color: {MAROON}; font-weight: 600; }}
    a {{ color: {MAROON}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Header — DSI logo + program wordmark
# ─────────────────────────────────────────────────────────────────────────────
logo_b64 = base64.b64encode((ASSETS / "dsi_logo.svg").read_bytes()).decode()

st.markdown(
    f"""
    <div class="uc-header">
        <img class="logo" src="data:image/svg+xml;base64,{logo_b64}" alt="UChicago Data Science Institute" />
        <div>
            <p class="uc-title">MS in Applied Data Science Assistant</p>
            <p class="uc-sub">Ask me anything about the University of Chicago MS-ADS program.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about the MS-ADS program..."):

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, retrieved_docs = get_answer(user_input)  # ← use your existing pipeline
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("📚 View retrieved sources"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Source {i+1}: {doc.metadata.get('section', 'Unknown')}**")
            st.caption(doc.metadata.get("url", ""))
            st.write(doc.page_content[:300] + "...")
            st.divider()
