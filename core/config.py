import os
from dotenv import load_dotenv

load_dotenv()

def get_google_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    if hasattr(api_key, 'get_secret_value'):
        return api_key.get_secret_value()
    else:
        return str(api_key)

GOOGLE_API_KEY = get_google_api_key()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
