import streamlit as st
from config import Config
from app_logger import AppLogger
import logging
from rag_pipeline import RagPipline
logger = AppLogger('streamlit', logging.INFO).setupLogger()

config = Config()
ragPipeline = RagPipline()
api_key = config.GEMINI_API_KEY
logger.info("App started")
logger.info(f"api key {api_key}")

st.set_page_config(layout="wide")
st.title("PDF Gen AI Assistant")

upload_file = st.file_uploader("Attach PDF File", type=['pdf'])

if st.button("Upload") and upload_file:
    logger.info(upload_file)
    ragPipeline.loadPDFDoc(upload_file)

query = st.text_input("Write your query", placeholder="Ask your query") 
if st.button("ASK") and query:
    with st.spinner("Fetching..."):
        result = ragPipeline.retriever(query)
        st.subheader("🔍 Answer:")
        st.write(result["result"])
        
    