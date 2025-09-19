# WhatsApp Chatbot with FastAPI, Twilio, and LangChain

A production-ready WhatsApp chatbot that uses FastAPI as the web framework, Twilio for WhatsApp integration, and LangChain with Google's Gemini LLM to handle user messages. The bot includes conversation memory and multimodal image processing capabilities.

## Features

- 🤖 **AI-Powered Responses**: Uses Google's Gemini models for intelligent conversation
- 💬 **Conversation Memory**: Remembers the last 5 conversation turns with each user
- 🖼️ **Image Processing**: Analyzes and describes images using Gemini 1.5 multimodal
- 🎯 **Motivational Coach Personality**: Provides supportive, encouraging responses
- 🐳 **Docker Ready**: Fully containerized for easy deployment
- 📱 **WhatsApp Integration**: Seamless Twilio webhook integration

## Prerequisites

- Python 3.11+
- Google API key (for Gemini)
- Twilio account with WhatsApp Sandbox or Business API access
- Docker (optional, for containerized deployment)

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
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at `http://localhost:8000`

### 5. Twilio Webhook Configuration

1. In your Twilio Console, go to your WhatsApp Sandbox or Business API
2. Set the webhook URL to: `https://your-domain.com/webhook`
3. For local testing, use ngrok or similar tunneling service:
   ```bash
   ngrok http 8000
   ```
   Then use the ngrok URL: `https://your-ngrok-url.ngrok.io/webhook`

## Docker Deployment

### Build the Docker Image

```bash
docker build -t whatsapp-chatbot .
```

### Run the Container

```bash
docker run -d \
  --name whatsapp-chatbot \
  -p 8000:8000 \
  --env-file .env \
  whatsapp-chatbot
```

### Docker Compose (Alternative)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  whatsapp-chatbot:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

Then run:
```bash
docker-compose up -d
```

## API Endpoints

### Webhook Endpoint
- **POST** `/webhook` - Twilio webhook for incoming WhatsApp messages

### Debug Endpoints
- **GET** `/` - Health check
- **GET** `/memory/{phone_number}` - View conversation memory for a user
- **DELETE** `/memory/{phone_number}` - Clear conversation memory for a user

## How It Works

1. **Message Reception**: Twilio sends incoming WhatsApp messages to the `/webhook` endpoint
2. **Message Processing**: 
   - For images: Uses Gemini 1.5 to analyze and describe the image
   - For text: Uses conversation memory and GPT model to generate responses
3. **Memory Management**: Stores conversation history per user phone number
4. **Response Generation**: Formats responses as TwiML and sends back via Twilio

## Conversation Memory

The bot maintains conversation memory using LangChain's `ConversationBufferMemory`:
- Stores last 5 conversation turns per user
- Keyed by user's phone number (`From` field)
- Automatically manages context for coherent conversations

## Image Processing

When users send images:
- Bot detects image attachments via `NumMedia` parameter
- Downloads image from Twilio's `MediaUrl0`
- Uses GPT-4 Vision to generate detailed descriptions
- Provides motivational coaching context around the image

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google API key (Gemini) | Yes |
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID | Yes |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token | Yes |
| `GEMINI_MODEL` | Gemini model to use (default: gemini-1.5-flash) | No |

## Troubleshooting

### Common Issues

1. **Webhook not receiving messages**
   - Verify Twilio webhook URL is correctly configured
   - Check that your server is accessible from the internet
   - Ensure HTTPS is used (required by Twilio)

2. **Gemini API errors**
   - Verify your `GOOGLE_API_KEY` is correct and has sufficient quota
   - Check rate limits and usage quotas in your Google AI Studio project

3. **Memory issues**
   - Use debug endpoints to inspect conversation memory
   - Clear memory if needed using DELETE endpoint

### Logs

The application logs all incoming messages and errors. Check the console output for debugging information.

## Security Considerations

- Never commit `.env` file to version control
- Use environment variables for all sensitive data
- Consider implementing rate limiting for production use
- Validate and sanitize all incoming webhook data

## Production Deployment

For production deployment:

1. Use a proper web server (nginx) as reverse proxy
2. Implement proper logging and monitoring
3. Set up SSL certificates
4. Configure proper backup strategies for conversation memory
5. Implement rate limiting and abuse prevention
6. Use a proper database for persistent memory storage

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review Twilio and Google Gemini documentation
- Open an issue in the repository
