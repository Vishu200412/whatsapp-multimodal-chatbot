import base64, tempfile, logging, time, os, requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm_module.llm import llm
from llm_module.embedding import embeddings
from core.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

def process_document(media_url, phone_number, rag_store):
    try:
        if embeddings is None:
            logger.error("No embedding model available")
            return False, "I'm sorry, the document processing system is not available right now. Please install sentence-transformers or scikit-learn for RAG functionality."
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            logger.error("Twilio credentials not found in environment variables.")
            return False, "I'm sorry, I cannot access the document because my Twilio credentials are not configured correctly."

        auth_header = base64.b64encode(f'{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}'.encode()).decode()
        headers = {'Authorization': f'Basic {auth_header}'}
        try:
            pdf_response = requests.get(media_url, headers=headers, timeout=30)
            pdf_response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("Timeout downloading PDF from Twilio")
            return False, "The document download timed out. Please try sending the document again."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF: {e}")
            return False, "I couldn't download the document. Please try sending it again."
        if len(pdf_response.content) > 10 * 1024 * 1024:
            return False, "The document is too large. Please send a PDF smaller than 10MB."
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_response.content)
            temp_file_path = temp_file.name
        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            if not documents:
                return False, "I couldn't extract any text from the PDF. Please make sure the document contains readable text."
            total_text = " ".join([doc.page_content for doc in documents]).strip()
            if len(total_text) < 50:
                return False, "The document appears to have very little text content. Please ensure it's a text-based PDF."
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split document into {len(chunks)} chunks")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Creating embeddings for {len(chunks)} chunks... (attempt {attempt + 1})")
                    batch_size = 5
                    vectorstore = None
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i + batch_size]
                        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                        if vectorstore is None:
                            vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                        else:
                            batch_vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                            vectorstore.merge_from(batch_vectorstore)
                        time.sleep(0.5)
                    logger.info("Successfully created vector store")
                    break
                except Exception as embedding_error:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(embedding_error)}")
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed: {str(embedding_error)}")
                        return False, "I'm sorry, I encountered an error while processing the document embeddings. This might be due to network issues or API limits. Please try again later."
                    time.sleep(2 ** attempt)
            try:
                summary_prompt = f"""
                Please provide a brief summary of this document in 2-3 sentences. 
                Focus on the main topic, purpose, and key information.
                Document content (first 1500 characters):
                {documents[0].page_content[:1500]}
                """
                summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
                document_summary = summary_response.content.strip()
            except Exception as summary_error:
                logger.warning(f"Failed to generate summary: {summary_error}")
                document_summary = f"Document with {len(chunks)} sections processed successfully."
            rag_store[phone_number] = {
                "vectorstore": vectorstore,
                "summary": document_summary
            }
            logger.info(f"Successfully processed document for {phone_number}")
            return True, f"✅ Thank you! I've successfully processed your document and can now answer questions about it.\n\n📄 Summary: {document_summary}"
        finally:
            try: os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False, "I'm sorry, I encountered an unexpected error while processing the document. Please try again or check if the PDF is valid."

def get_rag_response(user_question, vectorstore, conversation_chain):
    try:
        docs = vectorstore.similarity_search(user_question, k=2)
        context = "\n\n".join([doc.page_content for doc in docs])
        augmented_prompt = f"""
        You are a supportive, motivational life coach who helps people achieve their goals and overcome challenges. 
        You're empathetic, encouraging, and provide practical advice while maintaining a positive, uplifting tone.
        Use the following document context to answer the user's question accurately and helpfully:
        Document Context:
        {context}
        User's Question: {user_question}
        Please provide a helpful response based on the document context while maintaining your motivational coaching personality. 
        If the document doesn't contain relevant information, let the user know and offer to help in other ways.
        Keep your response under 200 words.
        """
        response = conversation_chain.predict(input=augmented_prompt)
        return response
    except Exception as e:
        logger.error(f"Error generating RAG response: {str(e)}")
        return "I'm having trouble accessing the document information right now. Please try again!"
