from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from langchain_google_genai import GoogleGenerativeAI
load_dotenv()
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LLM_CONFIG = {
        "gemini" : {
            "class" : GoogleGenerativeAI,
            "model" : "gemini-1.5-flash",
            "api_key" : GEMINI_API_KEY,
            "api_key_name" : 'google_api_key'
        },
        "openai" : {
            "class" : ChatOpenAI,
            "model" : "gpt-4o-mini",
            "api_key" : os.getenv('OPENAI_API_KEY'),
            "api_key_name" : 'api_key'
        },
        "grok" : {
            "class" : ChatXAI,
            "model" : "grok-4",
            "api_key" : os.getenv('GROK_API_KEY'),
            "api_key_name" : 'api_key'
        }
    }