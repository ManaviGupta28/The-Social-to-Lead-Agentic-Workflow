# AutoStream AI Agent

A production-ready Conversational AI Agent for AutoStream, a SaaS video editing platform. This agent uses **LangGraph** for state management, **RAG** (Retrieval-Augmented Generation) for accurate knowledge retrieval, and **intent detection** to identify high-value leads.

## 🚀 Features

- **Intent Classification**: Detects user intent (greeting, inquiry, high-intent lead)
- **RAG-Powered Responses**: Answers questions accurately using local knowledge base
- **Lead Capture**: Collects user information (name, email, platform) when high intent is detected
- **Tool Execution**: Triggers backend actions (mock lead capture API)
- **State Management**: Maintains conversation context across multiple turns
- **Webhook Interface**: FastAPI server ready for WhatsApp/Slack integration

## 📋 Prerequisites

- Python 3.9 or higher
- Google Gemini API key (free tier) - [Get it here](https://makersuite.google.com/app/apikey)

## 🛠️ Setup Instructions

### 1. Clone or Download the Project

```bash
cd "c:\Users\Manavi Gupta\Downloads\rag"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 5. Run the Application

```bash
python main.py
```

The server will start on `http://localhost:8000`

## 📡 API Usage

### Send a Message

```bash
curl -X POST "http://localhost:8000/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hi, how much is the Pro plan?",
    "thread_id": "user123"
  }'
```

### Response Format

```json
{
  "response": "The Pro plan is $79/month and includes...",
  "thread_id": "user123",
  "intent": "inquiry",
  "status": "success"
}
```

## 🏗️ Architecture Explanation

### Why LangGraph Over AutoGen?

I chose **LangGraph** for this project because:

1. **Explicit State Management**: LangGraph provides a clear state schema (TypedDict) that makes conversation flow transparent and debuggable. Unlike AutoGen's implicit message passing, LangGraph's state is structured and type-safe.

2. **Visual Graph Structure**: The conversation flow is modeled as a directed graph with nodes (intent classifier, RAG, lead capture) and conditional edges. This makes it easy to visualize and modify the conversation logic.

3. **Better Control Flow**: With LangGraph, I can implement sophisticated routing logic based on intent and state. The graph structure allows branching (e.g., inquiry → RAG vs high-intent → lead capture) that's harder to express in AutoGen's agent-to-agent model.

4. **Native Tool Integration**: LangGraph seamlessly integrates tool calling (like `mock_lead_capture`) as graph nodes, making it explicit when and how tools are invoked.

5. **Production Ready**: LangGraph's checkpointer system (using `MemorySaver`) enables stateful conversations across multiple API calls, critical for webhook-based deployments.

### State Management Approach

The agent uses **LangGraph's StateGraph** with a custom `AgentState` schema:

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]          # Full conversation history
    intent: str                          # Current user intent
    lead_info: Dict[str, Optional[str]]  # Collected lead data
    next_action: str                     # Routing control
    waiting_for: Optional[str]           # Expected user input
```

**Persistence**: State is maintained using `MemorySaver` checkpointer, which stores conversation state in memory keyed by `thread_id`. Each user session (simulated by phone number in WhatsApp) has isolated state.

**Conversation Flow**:
1. User message → Intent Classifier (detects greeting/inquiry/high-intent)
2. Based on intent → Route to appropriate node:
   - Greeting → Friendly welcome
   - Inquiry → RAG retrieval + LLM response
   - High-intent → Lead capture flow
3. Lead capture → Collect name, email, platform sequentially
4. All info collected → Execute `mock_lead_capture` tool

### RAG Pipeline Design

The RAG system uses:
- **FAISS** vector store for fast similarity search
- **Google Embeddings** (`models/embedding-001`) for semantic understanding
- **Knowledge base** structured as JSON with pricing, policies, and FAQs
- **Retrieval function** that fetches top-3 relevant contexts for each query

This ensures the agent provides accurate, up-to-date information without hallucination.

## 📱 WhatsApp Webhook Integration

To deploy this agent on **WhatsApp** using webhooks:

### Option 1: Meta Cloud API (Recommended)

1. **Create a Meta Developer Account**
   - Go to [Meta for Developers](https://developers.facebook.com/)
   - Create a new app with WhatsApp product

2. **Set Up Webhook**
   - Deploy this FastAPI app to a cloud server (e.g., AWS, Heroku, Railway)
   - Ensure it's accessible via HTTPS (use ngrok for testing)
   - In Meta dashboard, set webhook URL to `https://your-domain.com/webhook`

3. **Configure Webhook Handler**
   - Meta sends POST requests with format:
     ```json
     {
       "messages": [{
         "from": "1234567890",
         "text": {"body": "Hi, how much is the Pro plan?"}
       }]
     }
     ```
   - Modify `main.py` to parse Meta's format:
     ```python
     @app.post("/webhook")
     async def whatsapp_webhook(request: Request):
         data = await request.json()
         phone = data["messages"][0]["from"]
         message = data["messages"][0]["text"]["body"]
         
         # Use phone number as thread_id
         response = await webhook(WebhookRequest(
             message=message,
             thread_id=phone
         ))
         
         # Send response back to WhatsApp using Meta API
         send_whatsapp_message(phone, response.response)
     ```

4. **Handle Verification**
   - Meta requires a GET endpoint for webhook verification
   - Add to `main.py`:
     ```python
     @app.get("/webhook")
     async def verify_webhook(request: Request):
         mode = request.query_params.get("hub.mode")
         token = request.query_params.get("hub.verify_token")
         challenge = request.query_params.get("hub.challenge")
         
         if mode == "subscribe" and token == "YOUR_VERIFY_TOKEN":
             return int(challenge)
         return HTTPException(403)
     ```

### Option 2: Twilio WhatsApp API

1. **Create Twilio Account** at [twilio.com](https://www.twilio.com/)
2. **Enable WhatsApp Sandbox** (free for testing)
3. **Set Webhook URL** to your FastAPI endpoint
4. Twilio sends requests as:
   ```
   POST /webhook
   From=whatsapp:+1234567890
   Body=Hi, how much is the Pro plan?
   ```

### Deployment Checklist

- [ ] Deploy FastAPI app to cloud (Heroku/Railway/AWS)
- [ ] Enable HTTPS (required by Meta/Twilio)
- [ ] Set up environment variables on server
- [ ] Configure webhook URL in Meta/Twilio dashboard
- [ ] Test with WhatsApp sandbox
- [ ] Monitor logs for errors
- [ ] Scale with Redis/PostgreSQL for production state storage

## 📂 Project Structure

```
rag/
├── agent/
│   ├── __init__.py
│   ├── state.py          # State schema definition
│   ├── tools.py          # Lead capture tool
│   ├── nodes.py          # Graph nodes (intent, RAG, lead)
│   └── graph.py          # LangGraph workflow
├── rag/
│   ├── __init__.py
│   ├── knowledge_base.json  # Product knowledge
│   └── retriever.py      # RAG pipeline
├── main.py               # FastAPI webhook server
├── requirements.txt      # Dependencies
├── .env.example         # Environment template
└── README.md            # This file
```

## 🧪 Testing the Agent

### 6-Turn Conversation Test

```python
import requests

BASE_URL = "http://localhost:8000"
THREAD_ID = "test_user_001"

def send_message(message):
    response = requests.post(
        f"{BASE_URL}/webhook",
        json={"message": message, "thread_id": THREAD_ID}
    )
    return response.json()

# Turn 1: Pricing inquiry
print(send_message("Hi, how much is the Pro plan?"))

# Turn 2: High-intent signal
print(send_message("Cool, I want to sign up for my YouTube channel."))

# Turn 3: Provide name
print(send_message("My name is John Doe"))

# Turn 4: Provide email
print(send_message("john@example.com"))

# Turn 5: Provide platform
print(send_message("YouTube"))

# Turn 6: Verify tool execution (check console logs)
```

Expected console output on Turn 5/6:
```
🎯 Lead captured successfully: John Doe, john@example.com, YouTube
   Platform: YouTube
   Email: john@example.com
   Name: John Doe
```

## 🔧 Troubleshooting

### API Key Error
If you see `AuthenticationError`, ensure:
- `.env` file exists with valid `GOOGLE_API_KEY`
- API key is active on [Google AI Studio](https://makersuite.google.com/)

### Import Errors
Run: `pip install --upgrade -r requirements.txt`

### FAISS Installation Issues (Windows)
If `faiss-cpu` fails, try:
```bash
pip install faiss-cpu==1.7.4 --force-reinstall
```

## 📝 License

This project is created for the ServiceHive ML Intern assignment.

## 🤝 Author

Created for the **AutoStream Social-to-Lead Agentic Workflow** assignment.

---

**Note**: This agent uses Google Gemini 1.5 Flash (free tier). For production, consider rate limiting and error handling strategies.
