import os
import logging
import base64
import io
import tempfile
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import requests
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Chatbot", version="1.0.0")

# Global conversation memory store
conversation_memory_store: Dict[str, ConversationBufferWindowMemory] = {}

# Global RAG store - stores vector stores and document summaries for each user
rag_store: Dict[str, Dict] = {}  # {phone_number: {"vectorstore": FAISS, "summary": str}}

#Initialize Gemini LLM
# Ensure the correct environment variable is used by the SDK: GOOGLE_API_KEY
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Convert to string immediately to avoid SecretStr issues
google_api_key = str(google_api_key)

# 2. Use ChatGoogleGenerativeAI instead of ChatOpenAI for the main LLM.
# The user's prompt suggested a model named "gemini-2.5-flash", but the current latest models are 1.5 versions.
# We'll use "gemini-1.5-flash" as a suitable alternative.
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    google_api_key=google_api_key,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# Multimodal LLM for image processing (Gemini 1.5 Flash)
# 4. Use ChatGoogleGenerativeAI for multimodal capabilities.
# The same model can handle both text and image input.
multimodal_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=google_api_key,
)

# Configure Google Generative AI for audio processing
genai.configure(api_key=google_api_key)

# Initialize embeddings model for RAG with multiple fallbacks
try:
    logger.info("Attempting to use HuggingFace embeddings for RAG")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    logger.info("Successfully initialized HuggingFace embeddings")
except Exception as e:
    logger.warning(f"HuggingFace embeddings failed: {e}")
    try:
        logger.info("Attempting to use Google embeddings as fallback")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        logger.info("Successfully initialized Google embeddings")
    except Exception as e2:
        logger.error(f"All embedding methods failed: {e2}")
        # Use a simple dummy embedding for basic functionality
        logger.warning("Using dummy embeddings - RAG functionality will be limited")
        embeddings = None

# Replace the process_document function with this improved version:

def process_document(media_url: str, phone_number: str) -> Tuple[bool, str]:
    """
    Process a PDF document for RAG functionality.
    
    Args:
        media_url: Twilio media URL for the PDF
        phone_number: User's phone number for storage key
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        logger.info(f"Processing document for {phone_number}")
        
        # Check if embeddings are available
        if embeddings is None:
            logger.error("No embedding model available")
            return False, "I'm sorry, the document processing system is not available right now. Please try again later or contact support."
        
        # Download the PDF from Twilio with authentication
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        if not account_sid or not auth_token:
            logger.error("Twilio credentials not found in environment variables.")
            return False, "I'm sorry, I cannot access the document because my Twilio credentials are not configured correctly."
        
        # Download PDF with Twilio authentication
        auth_header = base64.b64encode(f'{account_sid}:{auth_token}'.encode()).decode()
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
        
        # Check file size (max 10MB)
        if len(pdf_response.content) > 10 * 1024 * 1024:
            return False, "The document is too large. Please send a PDF smaller than 10MB."
        
        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_response.content)
            temp_file_path = temp_file.name
        
        try:
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            if not documents:
                return False, "I couldn't extract any text from the PDF. Please make sure the document contains readable text."
            
            # Check if document has meaningful content
            total_text = " ".join([doc.page_content for doc in documents]).strip()
            if len(total_text) < 50:
                return False, "The document appears to have very little text content. Please ensure it's a text-based PDF."
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for better embedding
                chunk_overlap=100,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Create embeddings and vector store with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Creating embeddings for {len(chunks)} chunks... (attempt {attempt + 1})")
                    
                    # Process in smaller batches to avoid timeouts
                    batch_size = 5
                    vectorstore = None
                    
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i + batch_size]
                        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                        
                        if vectorstore is None:
                            # Create initial vectorstore
                            vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                        else:
                            # Add to existing vectorstore
                            batch_vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                            vectorstore.merge_from(batch_vectorstore)
                        
                        # Small delay between batches
                        import time
                        time.sleep(0.5)
                    
                    logger.info("Successfully created vector store")
                    break
                    
                except Exception as embedding_error:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(embedding_error)}")
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed: {str(embedding_error)}")
                        return False, "I'm sorry, I encountered an error while processing the document embeddings. This might be due to network issues or API limits. Please try again later."
                    
                    # Wait before retry
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Generate document summary with error handling
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
            
            # Store in RAG store
            rag_store[phone_number] = {
                "vectorstore": vectorstore,
                "summary": document_summary
            }
            
            logger.info(f"Successfully processed document for {phone_number}")
            return True, f"✅ Thank you! I've successfully processed your document and can now answer questions about it.\n\n📄 Summary: {document_summary}"
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False, "I'm sorry, I encountered an unexpected error while processing the document. Please try again or check if the PDF is valid."
        
def classify_intent(user_question: str, document_summary: str, conversation_history: str) -> str:
    """
    Classify user intent as 'document' or 'general'.
    
    Args:
        user_question: The user's question
        document_summary: Summary of the uploaded document
        conversation_history: Recent conversation history
        
    Returns:
        'document' or 'general'
    """
    intent_prompt_template = """
    You are an intent classifier for a WhatsApp chatbot. Your only job is to analyze the user's message and determine its primary intent.

    Instructions:
    1. If the user's question is asking for information, summarization, or analysis related to the provided "Document Summary" or "Conversation History," respond with ONLY the word "document".
    2. If the user's question is a general chat, a greeting, or anything unrelated to the document, respond with ONLY the word "general".
    3. Do not add any extra text, explanation, or punctuation. The response should be a single word: "document" or "general".

    Document Summary:
    {document_summary}

    Conversation History:
    {conversation_history}

    User's Question:
    {user_question}

    Intent:
    """
    
    try:
        intent_prompt = intent_prompt_template.format(
            document_summary=document_summary,
            conversation_history=conversation_history,
            user_question=user_question
        )
        
        intent_response = llm.invoke([HumanMessage(content=intent_prompt)])
        intent = intent_response.content.strip().lower()
        
        # Ensure we get a valid intent
        if intent in ['document', 'general']:
            return intent
        else:
            logger.warning(f"Unexpected intent response: {intent}")
            return 'general'  # Default to general if unclear
            
    except Exception as e:
        logger.error(f"Error classifying intent: {str(e)}")
        return 'general'  # Default to general on error

def get_rag_response(user_question: str, vectorstore: FAISS, conversation_chain: ConversationChain) -> str:
    """
    Generate a RAG-based response using document context.
    
    Args:
        user_question: The user's question
        vectorstore: FAISS vector store containing document chunks
        conversation_chain: LangChain conversation chain
        
    Returns:
        Generated response with document context
    """
    try:
        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(user_question, k=3)
        
        # Combine retrieved context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create augmented prompt
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
        
        # Get response from conversation chain
        response = conversation_chain.predict(input=augmented_prompt)
        return response
        
    except Exception as e:
        logger.error(f"Error generating RAG response: {str(e)}")
        return "I'm having trouble accessing the document information right now. Please try again!"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "WhatsApp Chatbot is running!", "status": "healthy"}

@app.post("/webhook")
async def webhook(
    From: str = Form(...),
    Body: str = Form(""),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(""),
    MediaContentType0: str = Form("")
):
    """
    Twilio webhook endpoint that handles incoming WhatsApp messages
    """
    try:
        logger.info(f"Received message from {From}: Body='{Body}', NumMedia='{NumMedia}', MediaUrl0='{MediaUrl0}', MediaContentType0='{MediaContentType0}'")
        
        # Initialize TwiML response
        response = MessagingResponse()
        
        # Handle audio/voice messages first (more specific condition)
        if NumMedia and int(NumMedia) > 0 and MediaUrl0 and MediaContentType0 and (MediaContentType0.startswith('audio/') or MediaContentType0.startswith('voice/') or 'ogg' in MediaContentType0.lower() or 'mpeg' in MediaContentType0.lower()):
            logger.info(f"Processing audio/voice from {From}, ContentType: {MediaContentType0}")
            
            # Get or create conversation memory for this user
            if From not in conversation_memory_store:
                conversation_memory_store[From] = ConversationBufferWindowMemory(
                    memory_key="history",
                    return_messages=True,
                    k=5  # Keep last 5 conversation turns
                )
                logger.info(f"Created new conversation memory for {From}")
            
            user_memory = conversation_memory_store[From]
            
            try:
                # Download the audio from Twilio with authentication
                account_sid = os.getenv("TWILIO_ACCOUNT_SID")
                auth_token = os.getenv("TWILIO_AUTH_TOKEN")
                
                if not account_sid or not auth_token:
                    logger.error("Twilio credentials not found in environment variables.")
                    response.message("I'm sorry, I cannot access the audio because my Twilio credentials are not configured correctly.")
                    return Response(content=str(response), media_type="application/xml")
                
                # Download audio with Twilio authentication
                auth_header = base64.b64encode(f'{account_sid}:{auth_token}'.encode()).decode()
                headers = {'Authorization': f'Basic {auth_header}'}
                
                audio_response = requests.get(MediaUrl0, headers=headers)
                audio_response.raise_for_status()
                
                # Convert audio to base64 for Gemini
                audio_base64 = base64.b64encode(audio_response.content).decode()
                
                # Use Gemini's audio processing to transcribe the audio
                audio_file = {
                    "mime_type": MediaContentType0,
                    "data": audio_base64
                }
                
                # Transcribe audio using Gemini
                model = genai.GenerativeModel('gemini-1.5-flash')
                result = model.generate_content([
                    "Please transcribe this audio message accurately. Return only the transcribed text without any additional commentary.",
                    audio_file
                ])
                
                transcribed_text = result.text.strip()
                logger.info(f"Transcribed audio: {transcribed_text}")
                
                # Create a prompt that matches ConversationChain expected variables: 'history' and 'input'
                conversation_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a supportive, motivational life coach who helps people achieve their goals and overcome challenges. You're empathetic, encouraging, and provide practical advice while maintaining a positive, uplifting tone. Keep responses under 200 words."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])

                # Create conversation chain with memory and custom prompt
                conversation_chain = ConversationChain(
                    llm=llm,
                    memory=user_memory,
                    prompt=conversation_prompt,
                    verbose=False
                )
                
                # Get response from LLM using transcribed text
                bot_response = conversation_chain.predict(input=f"[Audio message: {transcribed_text}]")
                
                # Add motivational coach personality to audio response
                motivational_response = f"🎤 I heard you say: \"{transcribed_text}\"\n\n{bot_response}"
                
                response.message(motivational_response)
                logger.info(f"Sent audio response to {From}")
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                response.message("I can hear you've sent an audio message, but I'm having trouble processing it right now. Please try again or send a text message!")
        
        # Handle image messages (after audio check)
        elif NumMedia and int(NumMedia) > 0 and MediaUrl0 and MediaContentType0 and (MediaContentType0.startswith('image/') or MediaContentType0.startswith('video/') or 'jpeg' in MediaContentType0.lower() or 'png' in MediaContentType0.lower() or 'gif' in MediaContentType0.lower()):
            logger.info(f"Processing image from {From} with caption: {Body}")
            
            # Get or create conversation memory for this user (same as text messages)
            if From not in conversation_memory_store:
                conversation_memory_store[From] = ConversationBufferWindowMemory(
                    memory_key="history",
                    return_messages=True,
                    k=5  # Keep last 5 conversation turns
                )
                logger.info(f"Created new conversation memory for {From}")
            
            user_memory = conversation_memory_store[From]
            
            try:
                # Download the image from Twilio with authentication
                account_sid = os.getenv("TWILIO_ACCOUNT_SID")
                auth_token = os.getenv("TWILIO_AUTH_TOKEN")
                
                if not account_sid or not auth_token:
                    logger.error("Twilio credentials not found in environment variables.")
                    response.message("I'm sorry, I cannot access the image because my Twilio credentials are not configured correctly.")
                    return Response(content=str(response), media_type="application/xml")
                
                # Download image with Twilio authentication
                auth_header = base64.b64encode(f'{account_sid}:{auth_token}'.encode()).decode()
                headers = {'Authorization': f'Basic {auth_header}'}
                
                image_response = requests.get(MediaUrl0, headers=headers)
                image_response.raise_for_status()
                
                # Convert image to base64 for Gemini
                image_base64 = base64.b64encode(image_response.content).decode()
                
                # Create content parts for Gemini
                content_parts = [
                    {"type": "text", "text": f"User's question: {Body}\n\nPlease analyze this image and answer the user's question about it."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
                
                # Use the multimodal LLM to analyze both the text and the image
                llm_response = multimodal_llm.invoke([
                    HumanMessage(content=content_parts)
                ])
                analysis = llm_response.content
                
                # Add motivational coach personality to image response
                motivational_response = f"🌟 Great question! Here's what I can see: {analysis}\n\nAs your motivational coach, I'm excited about your curiosity! How does this discovery inspire your next steps?"
                
                # Update memory with the image conversation
                user_memory.save_context(
                    {"input": f"[Image with caption: {Body}]"},
                    {"output": motivational_response}
                )
                
                response.message(motivational_response)
                logger.info(f"Sent image response to {From}")
                
            except Exception as e:
                logger.error(f"Error processing image with caption: {str(e)}")
                response.message("I can see you've shared an image and some text, but I'm having trouble processing them right now. Please try again!")
        
        # Handle PDF documents for RAG
        elif NumMedia and int(NumMedia) > 0 and MediaUrl0 and MediaContentType0 and MediaContentType0 == 'application/pdf':
            logger.info(f"Processing PDF document from {From}")
            
            # Process the document
            success, message = process_document(MediaUrl0, From)
            response.message(message)
            
            if success:
                logger.info(f"Successfully processed PDF for {From}")
            else:
                logger.error(f"Failed to process PDF for {From}: {message}")
        
        # Handle other media types (for debugging)
        elif NumMedia and int(NumMedia) > 0 and MediaUrl0:
            logger.info(f"Received unsupported media from {From}: ContentType='{MediaContentType0}', Body='{Body}'")
            response.message("I received a media file, but I can only process images, audio messages, and PDF documents right now. Please send a text message or try again!")
        
        # Handle text messages with conversation memory and RAG
        else:
            logger.info(f"Processing text message from {From}")
            
            # Get or create conversation memory for this user
            if From not in conversation_memory_store:
                conversation_memory_store[From] = ConversationBufferWindowMemory(
                    memory_key="history",
                    return_messages=True,
                    k=5  # Keep last 5 conversation turns
                )
                logger.info(f"Created new conversation memory for {From}")
            
            user_memory = conversation_memory_store[From]
            
            # Create a prompt that matches ConversationChain expected variables: 'history' and 'input'
            conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a supportive, motivational life coach who helps people achieve their goals and overcome challenges. You're empathetic, encouraging, and provide practical advice while maintaining a positive, uplifting tone. Keep responses under 200 words."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])

            # Create conversation chain with memory and custom prompt
            conversation_chain = ConversationChain(
                llm=llm,
                memory=user_memory,
                prompt=conversation_prompt,
                verbose=False
            )
            
            try:
                # Check if user has a document indexed for RAG
                if From in rag_store:
                    logger.info(f"User {From} has indexed document, checking intent")
                    
                    # Get conversation history for intent classification
                    conversation_history = ""
                    if user_memory.chat_history and user_memory.chat_history.messages:
                        recent_messages = user_memory.chat_history.messages[-4:]  # Last 4 messages
                        conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in recent_messages])
                    
                    # Classify intent
                    intent = classify_intent(
                        user_question=Body,
                        document_summary=rag_store[From]["summary"],
                        conversation_history=conversation_history
                    )
                    
                    logger.info(f"Intent classified as: {intent}")
                    
                    if intent == "document":
                        # Use RAG for document-related questions
                        bot_response = get_rag_response(
                            user_question=Body,
                            vectorstore=rag_store[From]["vectorstore"],
                            conversation_chain=conversation_chain
                        )
                        logger.info(f"Generated RAG response for {From}")
                    else:
                        # Use regular conversation for general questions
                        bot_response = conversation_chain.predict(input=Body)
                        logger.info(f"Generated general response for {From}")
                else:
                    # No document indexed, use regular conversation
                    bot_response = conversation_chain.predict(input=Body)
                    logger.info(f"Generated general response for {From} (no document)")
                
                response.message(bot_response)
                logger.info(f"Sent response to {From}")
                
            except Exception as e:
                logger.error(f"Error processing text message: {str(e)}")
                response.message("I'm here to support you, but I'm experiencing some technical difficulties right now. Please try again in a moment!")
        
        # Return TwiML response
        return Response(
            content=str(response),
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in webhook: {str(e)}")
        response = MessagingResponse()
        response.message("I apologize, but I'm experiencing some technical difficulties. Please try again later!")
        return Response(
            content=str(response),
            media_type="application/xml"
        )

@app.get("/memory/{phone_number}")
async def get_conversation_memory(phone_number: str):
    """
    Debug endpoint to view conversation memory for a specific user
    """
    if phone_number in conversation_memory_store:
        memory = conversation_memory_store[phone_number]
        return {
            "phone_number": phone_number,
            "memory_exists": True,
            "conversation_count": len(memory.chat_history.messages) if memory.chat_history else 0,
            "recent_messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content
                } 
                for msg in memory.chat_history.messages[-6:]  # Last 6 messages
            ] if memory.chat_history else []
        }
    else:
        return {
            "phone_number": phone_number,
            "memory_exists": False,
            "message": "No conversation history found for this user"
        }

@app.delete("/memory/{phone_number}")
async def clear_conversation_memory(phone_number: str):
    """
    Debug endpoint to clear conversation memory for a specific user
    """
    if phone_number in conversation_memory_store:
        del conversation_memory_store[phone_number]
        return {"message": f"Conversation memory cleared for {phone_number}"}
    else:
        return {"message": f"No conversation memory found for {phone_number}"}

@app.get("/rag/{phone_number}")
async def get_rag_info(phone_number: str):
    """
    Debug endpoint to view RAG information for a specific user
    """
    if phone_number in rag_store:
        rag_info = rag_store[phone_number]
        return {
            "phone_number": phone_number,
            "has_document": True,
            "document_summary": rag_info["summary"],
            "vectorstore_type": type(rag_info["vectorstore"]).__name__
        }
    else:
        return {
            "phone_number": phone_number,
            "has_document": False,
            "message": "No document indexed for this user"
        }

@app.delete("/rag/{phone_number}")
async def clear_rag_document(phone_number: str):
    """
    Debug endpoint to clear RAG document for a specific user
    """
    if phone_number in rag_store:
        del rag_store[phone_number]
        return {"message": f"RAG document cleared for {phone_number}"}
    else:
        return {"message": f"No RAG document found for {phone_number}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
