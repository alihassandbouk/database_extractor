from helper import get_chain
import streamlit as st

st.title("Ask the database")
question = st.text_input("")
if question:
    chain = get_chain()
    ans = chain(question)
    st.write(ans["result"])