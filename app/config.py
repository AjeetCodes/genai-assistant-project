from dotenv import load_dotenv
import os
load_dotenv()
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))