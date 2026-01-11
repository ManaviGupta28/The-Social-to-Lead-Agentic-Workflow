"""
LangGraph Nodes for AutoStream Agent

Each node represents a processing step in the conversation flow:
- Intent classification
- RAG-based knowledge retrieval
- Lead information collection
- Tool execution
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from agent.tools import mock_lead_capture, extract_lead_info
from rag.retriever import get_retriever


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    convert_system_message_to_human=True
)


def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify user intent from the latest message
    
    Intents:
    - greeting: Casual conversation starter
    - inquiry: Product/pricing questions
    - high_intent: User wants to sign up/purchase
    - unknown: Cannot determine intent
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Create prompt for intent classification
    system_prompt = """You are an intent classifier for a SaaS sales agent.
Classify the user's message into one of these intents:

1. "greeting" - User is saying hi, hello, or casual greeting
2. "inquiry" - User is asking about product features, pricing, or policies
3. "high_intent" - User wants to sign up, purchase, try the product, or shows strong buying interest
4. "unknown" - Cannot determine intent

Respond with ONLY the intent label (greeting/inquiry/high_intent/unknown), nothing else.

User message: {message}
"""
    
    # Simple rule-based classification with LLM fallback
    message_lower = last_message.lower()
    
    # High intent signals
    high_intent_keywords = [
        "sign up", "want to try", "i want", "get started", 
        "purchase", "buy", "subscribe", "interested in",
        "ready to", "let's go", "sounds good", "i'll take"
    ]
    
    # Inquiry signals
    inquiry_keywords = [
        "how much", "price", "pricing", "cost", "plan", "feature",
        "what is", "tell me about", "do you", "can i", "support",
        "refund", "cancel", "trial"
    ]
    
    # Greeting signals
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon"]
    
    # Check for high intent first (highest priority)
    if any(keyword in message_lower for keyword in high_intent_keywords):
        intent = "high_intent"
    elif any(keyword in message_lower for keyword in inquiry_keywords):
        intent = "inquiry"
    elif any(keyword in message_lower for keyword in greeting_keywords) and len(message_lower.split()) < 10:
        intent = "greeting"
    else:
        # Use LLM for ambiguous cases
        try:
            response = llm.invoke([HumanMessage(content=system_prompt.format(message=last_message))])
            intent = response.content.strip().lower()
            if intent not in ["greeting", "inquiry", "high_intent", "unknown"]:
                intent = "inquiry"  # Default to inquiry
        except:
            intent = "inquiry"
    
    return {
        "intent": intent,
        "next_action": "route"
    }


def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Answer user questions using RAG from knowledge base
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Retrieve relevant context
    retriever = get_retriever()
    context = retriever.get_context(last_message, k=3)
    
    # Create RAG prompt
    system_prompt = f"""You are a helpful sales assistant for AutoStream, an automated video editing SaaS platform.

Use the following context to answer the user's question accurately and conversationally.
If the question is not covered in the context, politely say you don't have that information.

Context:
{context}

Guidelines:
- Be friendly and professional
- Highlight key benefits when discussing features
- If asked about pricing, mention both plans and their key differences
- Encourage users to ask follow-up questions
- Don't make up information not in the context

User question: {last_message}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=system_prompt)])
        ai_response = response.content
    except Exception as e:
        ai_response = "I apologize, I'm having trouble accessing information right now. Could you please try again?"
    
    return {
        "messages": [AIMessage(content=ai_response)],
        "next_action": "end"
    }


def lead_capture_node(state: AgentState) -> Dict[str, Any]:
    """
    Collect lead information step by step
    """
    messages = state["messages"]
    lead_info = state.get("lead_info", {"name": None, "email": None, "platform": None})
    waiting_for = state.get("waiting_for")
    last_message = messages[-1].content if messages else ""
    
    # If we just detected high intent, start collection
    if waiting_for is None:
        response = "That's great! I'd love to help you get started with the Pro plan. Can I get your name first?"
        return {
            "messages": [AIMessage(content=response)],
            "lead_info": lead_info,
            "waiting_for": "name",
            "next_action": "end"
        }
    
    # Extract information based on what we're waiting for
    extracted = extract_lead_info(last_message, waiting_for)
    lead_info.update(extracted)
    
    # Determine next step
    if not lead_info.get("name"):
        response = "Could you please provide your name?"
        next_waiting = "name"
    elif not lead_info.get("email"):
        response = f"Thanks, {lead_info['name']}! What's your email address?"
        next_waiting = "email"
    elif not lead_info.get("platform"):
        response = "Great! Which platform do you primarily create content for? (e.g., YouTube, Instagram, TikTok)"
        next_waiting = "platform"
    else:
        # All information collected, trigger tool
        next_waiting = None
        response = None  # Will be set by tool execution
    
    if next_waiting:
        return {
            "messages": [AIMessage(content=response)],
            "lead_info": lead_info,
            "waiting_for": next_waiting,
            "next_action": "end"
        }
    else:
        return {
            "lead_info": lead_info,
            "waiting_for": None,
            "next_action": "execute_tool"
        }


def tool_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute the lead capture tool when all info is collected
    """
    lead_info = state["lead_info"]
    
    # Call the mock lead capture function
    result = mock_lead_capture(
        name=lead_info["name"],
        email=lead_info["email"],
        platform=lead_info["platform"]
    )
    
    if result == "SUCCESS":
        response = f"Perfect! I've got you all set up, {lead_info['name']}. You'll receive an email at {lead_info['email']} with instructions to activate your Pro plan. Welcome to AutoStream! 🎉"
    else:
        response = "I'm sorry, there was an issue capturing your information. Please try again later."
    
    return {
        "messages": [AIMessage(content=response)],
        "next_action": "end"
    }


def greeting_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle casual greetings
    """
    response = "Hi there! 👋 I'm here to help you learn about AutoStream, our AI-powered video editing platform for content creators. What would you like to know?"
    
    return {
        "messages": [AIMessage(content=response)],
        "next_action": "end"
    }
