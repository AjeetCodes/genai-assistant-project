import streamlit as st
from config import Config
from app_logger import AppLogger
import logging
from rag_pipeline import RagPipline
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from config import Config
logger = AppLogger('streamlit', logging.INFO).setupLogger()

config = Config()
api_key = config.GEMINI_API_KEY
logger.info("App started")
logger.info(f"api key {api_key}")

st.set_page_config(layout="wide", page_title="Multi-Docs Upload", page_icon="📄")
# st.title("Gen AI Assistant")

st.markdown("""
    <style>
        .stApp {
            background-color: #f9f9f9;
        }
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-header h1 {
            color: #333;
        }
        .uploaded-file {
            font-size: 14px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <h1>🤖 Gemini Chatbot</h1>
    <p>LangChain + Streamlit + Gemini AI</p>
</div>
""", unsafe_allow_html=True)

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
# Show chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(f"🧑‍💻 **You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(f"🤖 **Gemini:** {msg.content}")
        
# query = st.text_input("Write your query", placeholder="Ask your query") 
prompt = st.chat_input("Type your message..")
if prompt:
    with st.spinner("Fetching..."):
        # Add user message
        st.chat_message("user").markdown(f"🧑‍💻 **You:** {prompt}")
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Build system prompt using tone and optional file
        system_prefix = f"You are a {selected_tone.lower()} assistant."
        # Add system prompt only if first message
        if len(st.session_state.messages) == 1:
            st.session_state.messages.insert(0, HumanMessage(content=system_prefix))
        
        result = ragPipeline.retriever(prompt)
        if 'result' in result:
            response = result["result"] 
            st.chat_message("assistant").markdown(f"🤖 **Gemini:** {response}")
            st.session_state.messages.append(AIMessage(content=response))
        else:
            st.warning(result['error'])
    