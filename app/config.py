from dotenv import load_dotenv
import os
load_dotenv()
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')