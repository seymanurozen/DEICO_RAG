import streamlit as st

from ragify import Ragify

rag_pipeline = Ragify(
    llm_name="",
)

st.text(
    rag_pipeline.generate_response(
        question="Sample Question"
    )
)