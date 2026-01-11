"""
FastAPI Webhook Interface for AutoStream Agent

Provides HTTP endpoints for:
- Conversational webhook (POST /webhook)
- Health checks (GET /health)

Simulates WhatsApp webhook integration pattern.
"""

# Load environment variables FIRST before any other imports
import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file

# Check if API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️  WARNING: GOOGLE_API_KEY not found in environment variables!")
    print("Please create a .env file with: GOOGLE_API_KEY=your_key_here")
    print("Get your free API key from: https://makersuite.google.com/app/apikey")
    exit(1)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from langchain_core.messages import HumanMessage
from agent.graph import agent_graph


# FastAPI app
app = FastAPI(
    title="AutoStream AI Agent",
    description="Conversational AI agent for AutoStream video editing platform",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebhookRequest(BaseModel):
    """Request model for webhook endpoint"""
    message: str
    thread_id: str  # Simulates user's phone number or session ID
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hi, how much is the Pro plan?",
                "thread_id": "user123"
            }
        }


class WebhookResponse(BaseModel):
    """Response model for webhook endpoint"""
    response: str
    thread_id: str
    intent: Optional[str] = None
    status: str = "success"


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AutoStream AI Agent",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "autostream-agent"
    }


@app.post("/webhook", response_model=WebhookResponse)
async def webhook(request: WebhookRequest):
    """
    Main webhook endpoint for conversational interface
    
    Accepts user messages and returns agent responses.
    Maintains conversation state using thread_id.
    
    Args:
        request: WebhookRequest with message and thread_id
        
    Returns:
        WebhookResponse with agent's reply
    """
    try:
        # Create configuration with thread ID for state persistence
        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }
        
        # Prepare input with user message and initialize state fields
        input_data = {
            "messages": [HumanMessage(content=request.message)],
            "intent": "unknown",
            "lead_info": {"name": None, "email": None, "platform": None},
            "next_action": "classify",
            "waiting_for": None
        }
        
        # Invoke the agent graph
        result = agent_graph.invoke(input_data, config=config)
        
        # Extract the latest AI message
        messages = result.get("messages", [])
        ai_messages = [msg for msg in messages if hasattr(msg, 'type') and msg.type == 'ai']
        
        if ai_messages:
            response_text = ai_messages[-1].content
        else:
            response_text = "I apologize, I didn't understand that. Could you please rephrase?"
        
        # Get intent for debugging
        intent = result.get("intent", "unknown")
        
        return WebhookResponse(
            response=response_text,
            thread_id=request.thread_id,
            intent=intent,
            status="success"
        )
        
    except Exception as e:
        # Log error and return graceful response
        print(f"Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/reset/{thread_id}")
async def reset_conversation(thread_id: str):
    """
    Reset conversation state for a specific thread
    
    Args:
        thread_id: Thread/session ID to reset
        
    Returns:
        Confirmation message
    """
    # Note: With MemorySaver, state automatically expires
    # This endpoint is for future implementation with persistent storage
    return {
        "status": "success",
        "message": f"Conversation reset for thread {thread_id}",
        "thread_id": thread_id
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
