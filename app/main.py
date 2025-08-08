import streamlit as st
from config import Config
from app_logger import AppLogger
import logging

logger = AppLogger('streamlit', logging.INFO).setupLogger()

config = Config()
api_key = config.GEMINI_API_KEY
logger.info("App started")
logger.info(f"api key {api_key}")

st.set_page_config(layout="wide")
st.title("PDF Gen AI Assistant")

upload_file = st.file_uploader("Attach PDF File", type=['pdf'])

if st.button("Upload") and upload_file:
    logger.info(upload_file)