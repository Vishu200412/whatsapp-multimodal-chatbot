import google.generativeai as genai
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import GOOGLE_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)
try:
    multimodal_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )
except Exception as e:
    logger.error(f"Failed to initialize multimodal LLM: {e}")
    multimodal_llm = None

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {e}")
    raise
