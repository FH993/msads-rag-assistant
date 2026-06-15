import streamlit as st
from rag_chain import get_answer  

st.set_page_config(
    page_title="UChicago MS-ADS Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 UChicago MS in Applied Data Science")
st.subheader("Ask me anything about the program!")

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