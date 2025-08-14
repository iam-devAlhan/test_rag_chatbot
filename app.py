import streamlit as st
st.set_page_config(page_title="Knowledge Chatbot")

st.title("📚 Your Personal Knowledge Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask something from your content:")

if query:
    from bot import get_answer  # optional helper

    response = get_answer(query)
    st.session_state.history.append((query, response))

for q, r in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {r}")