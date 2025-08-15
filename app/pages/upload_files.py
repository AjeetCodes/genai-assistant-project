import streamlit as st
from config import Config
from app_logger import AppLogger
import logging
from rag_pipeline import RagPipline
import time
logger = AppLogger('streamlit', logging.INFO).setupLogger()

config = Config()
# ragPipeline = RagPipline()


st.set_page_config(layout="wide", page_title="Multi-PDF Upload", page_icon="📄")
st.title("PDF Gen AI Assistant")
with st.sidebar:
    st.header("⚙️ Settings")
    selected_tone = st.selectbox("Assistant Tone", ["Friendly", "Professional", "Funny", "Technical"])
    selectedLlmModel = st.selectbox("Select LLM Model", Config().LLM_CONFIG.keys())

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

ragPipeline = RagPipline(selectedLlmModel)

upload_file = st.file_uploader("Upload one or more PDFs", type=['pdf', 'docx'], accept_multiple_files=True)

if st.button("Upload") and upload_file:
    logger.info(upload_file)
    total_files = len(upload_file)
    st.write(f"📂 {total_files} file(s) uploaded")

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_texts = []
    for i, file in enumerate(upload_file):
        status_text.text(f"Processing: {file.name} ({i+1}/{total_files})")
        logger.info(f"file {file}")
        # Simulate processing delay
        time.sleep(0.5)
        file_ext = file.name.split(".")[-1].lower()
        logger.info(f"file ext {file_ext}")
        if file_ext == "pdf":
            ragPipeline.loadPDFDoc(file)
        elif file_ext == "docx":
            ragPipeline.loadDocx(file)
        # Update progress bar
        progress_bar.progress((i + 1) / total_files)
    status_text.text("✅ All PDFs processed!")
    st.success(f"Extracted text from {total_files} PDFs successfully.")

query = st.text_input("Write your query", placeholder="Ask your query") 
if st.button("ASK") and query:
    with st.spinner("Fetching..."):
        result = ragPipeline.retriever(query)
        if 'result' in result:
            st.subheader("🔍 Answer:")
            st.write(result["result"])
        else:
            st.warning(result['error'])
    