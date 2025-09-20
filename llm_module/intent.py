import logging
from llm import llm
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

def classify_intent(user_question, document_summary, conversation_history):
    intent_prompt_template = """
    You are an intent classifier for a WhatsApp chatbot. Your only job is to analyze the user's message and determine its primary intent.

    Instructions:
    1. If the user's question is asking for information, summarization, or analysis related to the provided "Document Summary",  respond with ONLY the word "document".
    2. If the user's question is a related to "Conversation History", general chat, a greeting, or anything unrelated to the document, respond with ONLY the word "general".
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
        if intent in ['document', 'general']:
            logger.info(f"Intent classified: {intent}")
            return intent
        else:
            logger.warning(f"Unexpected intent response: {intent}")
            return 'general'
    except Exception as e:
        logger.error(f"Error classifying intent: {str(e)}")
        return 'general'
