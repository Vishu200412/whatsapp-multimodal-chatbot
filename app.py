import logging, base64, time, os, requests
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from core.config import GEMINI_MODEL
from llm_module.llm import llm, multimodal_llm
from llm_module.embedding import embeddings, embedding_model_name, embedding_success
from rag_module.rag import process_document, get_rag_response
from llm_module.intent import classify_intent
from core.utils import setup_logging
import google.generativeai as genai

logger = setup_logging()
app = FastAPI(title="WhatsApp Chatbot", version="1.0.0")

conversation_memory_store = {}
rag_store = {}

@app.get("/")
async def root():
    return {"message": "WhatsApp Chatbot is running!", "status": "healthy"}

@app.post("/webhook")
async def webhook(
    From: str = Form(...),
    Body: str = Form(""),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(""),
    MediaContentType0: str = Form("")
):
    try:
        logger.info(f"Received message from {From}: Body='{Body}', NumMedia='{NumMedia}', MediaUrl0='{MediaUrl0}', MediaContentType0='{MediaContentType0}'")
        response = MessagingResponse()
        # AUDIO
        if NumMedia and int(NumMedia) > 0 and MediaUrl0 and MediaContentType0 and (MediaContentType0.startswith('audio/') or MediaContentType0.startswith('voice/') or 'ogg' in MediaContentType0.lower() or 'mpeg' in MediaContentType0.lower()):
            if From not in conversation_memory_store:
                conversation_memory_store[From] = ConversationBufferWindowMemory(
                    memory_key="history", return_messages=True, k=7)
            user_memory = conversation_memory_store[From]
            try:
                from core.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
                if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
                    response.message("I'm sorry, I cannot access the audio because my Twilio credentials are not configured correctly.")
                    return Response(content=str(response), media_type="application/xml")
                auth_header = base64.b64encode(f'{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}'.encode()).decode()
                headers = {'Authorization': f'Basic {auth_header}'}
                audio_response = requests.get(MediaUrl0, headers=headers)
                audio_response.raise_for_status()
                audio_base64 = base64.b64encode(audio_response.content).decode()
                audio_file = {
                    "mime_type": MediaContentType0,
                    "data": audio_base64
                }
                model = genai.GenerativeModel('gemini-1.5-flash')
                result = model.generate_content([
                    "Please transcribe this audio message accurately. Return only the transcribed text without any additional commentary.",
                    audio_file
                ])
                transcribed_text = result.text.strip()
                conversation_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a supportive, motivational life coach ..."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
                conversation_chain = ConversationChain(
                    llm=llm,
                    memory=user_memory,
                    prompt=conversation_prompt,
                    verbose=False
                )
                bot_response = conversation_chain.predict(input=f"[Audio message: {transcribed_text}]")
                motivational_response = f"🎤 I heard you say: \"{transcribed_text}\"\n\n{bot_response}"
                response.message(motivational_response)
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                response.message("I can hear you've sent an audio message, but I'm having trouble processing it right now. Please try again or send a text message!")
        # IMAGE
        elif NumMedia and int(NumMedia) > 0 and MediaUrl0 and MediaContentType0 and (MediaContentType0.startswith('image/') or MediaContentType0.startswith('video/') or 'jpeg' in MediaContentType0.lower() or 'png' in MediaContentType0.lower() or 'gif' in MediaContentType0.lower()):
            if From not in conversation_memory_store:
                conversation_memory_store[From] = ConversationBufferWindowMemory(
                    memory_key="history", return_messages=True, k=7)
            user_memory = conversation_memory_store[From]
            try:
                from core.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
                if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
                    response.message("I'm sorry, I cannot access the image because my Twilio credentials are not configured correctly.")
                    return Response(content=str(response), media_type="application/xml")
                auth_header = base64.b64encode(f'{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}'.encode()).decode()
                headers = {'Authorization': f'Basic {auth_header}'}
                image_response = requests.get(MediaUrl0, headers=headers)
                image_response.raise_for_status()
                image_base64 = base64.b64encode(image_response.content).decode()
                content_parts = [
                    {"type": "text", "text": f"User's question: {Body}\n\nPlease analyze this image and answer the user's question about it."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
                llm_response = multimodal_llm.invoke([
                    HumanMessage(content=content_parts)
                ])
                analysis = llm_response.content
                motivational_response = f"🌟 Great question! Here's what I can see: {analysis}\n\nAs your motivational coach, I'm excited about your curiosity! How does this discovery inspire your next steps?"
                user_memory.save_context(
                    {"input": f"[Image with caption: {Body}]"},
                    {"output": motivational_response}
                )
                response.message(motivational_response)
            except Exception as e:
                logger.error(f"Error processing image with caption: {str(e)}")
                response.message("I can see you've shared an image and some text, but I'm having trouble processing them right now. Please try again!")
        # PDF DOCS
        elif NumMedia and int(NumMedia) > 0 and MediaUrl0 and MediaContentType0 and MediaContentType0 == 'application/pdf':
            success, message = process_document(MediaUrl0, From, rag_store)
            response.message(message)
        # OTHER MEDIA
        elif NumMedia and int(NumMedia) > 0 and MediaUrl0:
            response.message("I received a media file, but I can only process images, audio messages, and PDF documents right now. Please send a text message or try again!")
        # TEXT
        else:
            if From not in conversation_memory_store:
                conversation_memory_store[From] = ConversationBufferWindowMemory(
                    memory_key="history", return_messages=True, k=7)
            user_memory = conversation_memory_store[From]
            conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a supportive, motivational life coach ..., when answering the question if the inforation is not available in the document, look into the conversation history to answer the question and if the information is not available in the conversation history, answer the question as best as you can regardless of the history "),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            conversation_chain = ConversationChain(
                llm=llm,
                memory=user_memory,
                prompt=conversation_prompt,
                verbose=False
            )
            try:
                if From in rag_store:
                    conversation_history = ""
                    try:
                        if hasattr(user_memory, 'chat_memory') and user_memory.chat_memory and user_memory.chat_memory.messages:
                            recent_messages = user_memory.chat_memory.messages[-4:]
                            conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in recent_messages])
                        elif hasattr(user_memory, 'memory_buffer') and user_memory.memory_buffer:
                            conversation_history = user_memory.memory_buffer
                        elif hasattr(user_memory, 'buffer') and user_memory.buffer:
                            conversation_history = user_memory.buffer
                        else:
                            conversation_history = "No recent conversation history available."
                    except Exception as e:
                        logger.warning(f"Could not retrieve conversation history: {e}")
                        conversation_history = "No recent conversation history available."
                    intent = classify_intent(
                        user_question=Body,
                        document_summary=rag_store[From]["summary"],
                        conversation_history=conversation_history
                    )
                    if intent == "document":
                        bot_response = get_rag_response(
                            user_question=Body,
                            vectorstore=rag_store[From]["vectorstore"],
                            conversation_chain=conversation_chain
                        )
                    else:
                        bot_response = conversation_chain.predict(input=Body)
                else:
                    bot_response = conversation_chain.predict(input=Body)
                response.message(bot_response)
            except Exception as e:
                logger.error(f"Error processing text message: {str(e)}")
                response.message("I'm here to support you, but I'm experiencing some technical difficulties right now. Please try again in a moment!")
        return Response(content=str(response), media_type="application/xml")
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
    if phone_number in conversation_memory_store:
        memory = conversation_memory_store[phone_number]
        conversation_count = 0
        recent_messages = []
        try:
            if hasattr(memory, 'chat_memory') and memory.chat_memory and memory.chat_memory.messages:
                conversation_count = len(memory.chat_memory.messages)
                recent_messages = [
                    {
                        "type": msg.__class__.__name__,
                        "content": msg.content
                    } 
                    for msg in memory.chat_memory.messages[-6:]
                ]
            elif hasattr(memory, 'memory_buffer') and memory.memory_buffer:
                conversation_count = 1
                recent_messages = [{"type": "Buffer", "content": memory.memory_buffer}]
        except Exception as e:
            logger.warning(f"Could not retrieve memory details: {e}")
        return {
            "phone_number": phone_number,
            "memory_exists": True,
            "conversation_count": conversation_count,
            "recent_messages": recent_messages
        }
    else:
        return {
            "phone_number": phone_number,
            "memory_exists": False,
            "message": "No conversation history found for this user"
        }

@app.delete("/memory/{phone_number}")
async def clear_conversation_memory(phone_number: str):
    if phone_number in conversation_memory_store:
        del conversation_memory_store[phone_number]
        return {"message": f"Conversation memory cleared for {phone_number}"}
    else:
        return {"message": f"No conversation memory found for {phone_number}"}

@app.get("/rag/{phone_number}")
async def get_rag_info(phone_number: str):
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
    if phone_number in rag_store:
        del rag_store[phone_number]
        return {"message": f"RAG document cleared for {phone_number}"}
    else:
        return {"message": f"No RAG document found for {phone_number}"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "embedding_model": embedding_model_name,
        "has_embeddings": embeddings is not None,
        "active_conversations": len(conversation_memory_store),
        "active_documents": len(rag_store),
        "llm_model": GEMINI_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
