import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
embeddings = None
embedding_model_name = "unknown"

def initialize_huggingface_embeddings():
    global embeddings, embedding_model_name
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        embedding_model_name = "HuggingFace"
        return True
    except Exception as e:
        logger.warning(f"HuggingFace embeddings failed: {e}")
    return False

embedding_success = initialize_huggingface_embeddings()
