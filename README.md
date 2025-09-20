# WhatsApp Chatbot with FastAPI, Twilio, and LangChain

A WhatsApp chatbot that uses FastAPI as the web framework, Twilio for WhatsApp integration, and LangChain with Google's Gemini LLM to handle user messages. The bot includes conversation memory, multimodal image processing, and document-based question answering using RAG (Retrieval-Augmented Generation).

## Demo Video

[![Watch the demo](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID)

Click the image above to watch the demo video.

## Features

- 🤖 **AI-Powered Responses**: Uses Google's Gemini models for intelligent conversation  
- 💬 **Conversation Memory**: Remembers the last 5 conversation turns with each user  
- 🖼️ **Image Processing**: Analyzes and describes images using Gemini 1.5 multimodal  
- 📄 **Document Upload & RAG Q&A**: Users can upload PDFs; the bot indexes the content enabling question answering based on the document  
- 🎯 **Motivational Coach Personality**: Provides supportive, encouraging responses  
- 📱 **WhatsApp Integration**: Seamless Twilio webhook integration  

## Prerequisites

- Python 3.11+  
- Google API key (for Gemini)  
- Twilio account with WhatsApp Sandbox or Business API access  

## Setup Instructions

### 1. Clone and Navigate to Project

```bash
git clone <your-repo-url>
cd whatsapp-chatbot
```

### 2. Environment Configuration

Create a `.env` file in the project root with:

```env
GOOGLE_API_KEY=your_actual_google_api_key_here
GEMINI_MODEL=gemini-1.5-flash
TWILIO_ACCOUNT_SID=your_actual_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_actual_twilio_auth_token_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at `http://localhost:8000`.

### 5. Twilio Webhook Configuration

1. In your Twilio Console, go to your WhatsApp Sandbox or Business API  
2. Set the webhook URL to: `https://your-domain.com/webhook`  
3. For local testing, use ngrok or a similar tunneling service:  
   ```bash
   ngrok http 8000
   ```
   Then use the ngrok URL: `https://your-ngrok-url.ngrok.io/webhook`

## API Endpoints

### Webhook Endpoint

- **POST** `/webhook` - Twilio webhook for incoming WhatsApp messages  

### Debug Endpoints

- **GET** `/` - Health check  
- **GET** `/memory/{phone_number}` - View conversation memory for a user  
- **DELETE** `/memory/{phone_number}` - Clear conversation memory for a user  
- **GET** `/rag/{phone_number}` - View document RAG info for a user  
- **DELETE** `/rag/{phone_number}` - Clear document RAG data for a user  

## How It Works

1. **Message Reception**: Twilio sends incoming WhatsApp messages to the `/webhook` endpoint  
2. **Message Processing**:  
   - For images: Uses Gemini 1.5 multimodal to analyze and describe the image  
   - For PDFs: Downloads and processes documents to create vector embeddings and summaries  
   - For text: Uses conversation memory, document retrieval (if applicable), and Gemini LLM to generate responses  
3. **Memory Management**: Stores per-user conversation history keyed by phone number  
4. **RAG (Document Q&A)**: Enables users to ask questions about uploaded PDF documents. The bot uses similarity search on embedded chunks and generates answers based on document context with a motivational coaching tone  
5. **Response Generation**: Formats all responses as TwiML and sends them back via Twilio  

## Conversation Memory

The bot maintains conversation memory using LangChain's `ConversationBufferWindowMemory`:  

- Stores last 7 conversation turns per user  
- Keyed by user's phone number (the `From` field in WhatsApp)  
- Automatically manages context for coherent conversations  

## Document Upload and RAG Q&A

- Users can upload PDF documents via WhatsApp  
- The bot downloads and splits the document into chunks, then creates embeddings using HuggingFace embeddings  
- These vectors are stored in a FAISS vector store for quick similarity search  
- The bot generates a brief document summary using Gemini LLM  
- When users ask questions related to the uploaded document, the bot retrieves relevant document chunks and generates targeted answers based on document content  
- This process allows contextual and accurate Q&A about user-provided documents, making the chatbot highly versatile  

## Image Processing

- Detects image or video attachments sent in WhatsApp messages  
- Downloads the media via Twilio with authentication  
- Uses Gemini 1.5 multimodal LLM to analyze the image and answer user questions related to it  
- Responds with an encouraging, motivational coaching style  

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google API key (Gemini) | Yes |
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID | Yes |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token | Yes |
| `GEMINI_MODEL` | Gemini model to use (default: gemini-1.5-flash) | No |

## Troubleshooting

- **Webhook not receiving messages**: Verify webhook URL, server accessibility, and HTTPS usage  
- **Gemini API errors**: Check API key correctness and quota limits  
- **Memory issues**: Use debug endpoints to inspect and clear memory if needed  
- **Document processing errors**: Ensure PDF is text-based and under 10MB, retry on network issues  

## Security Considerations

- Never commit `.env` file to version control  
- Use environment variables for sensitive data  
- Validate all incoming webhook data  
- Consider rate limiting for production use  

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Add tests if applicable  
5. Submit a pull request  

## License

This project is licensed under the MIT License.