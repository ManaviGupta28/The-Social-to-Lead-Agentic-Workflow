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

# Add safety settings to avoid empty responses from filters during debugging
from langchain_google_genai import HarmCategory, HarmBlockThreshold

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7,
    convert_system_message_to_human=True,
    safety_settings=safety_settings
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
    # If we're already in lead capture mode, preserve intent and skip classification
    waiting_for = state.get("waiting_for")
    if waiting_for:
        # We're collecting lead info - don't change intent
        return {
            "intent": state.get("intent", "high_intent"),
            "next_action": "route"
        }
    
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


def _get_fallback_response(query: str, retriever) -> str:
    """
    Generate a fallback response using direct knowledge base access
    when LLM fails
    
    This function loads the knowledge base JSON directly and generates
    accurate responses for pricing/plan questions without relying on LLM.
    """
    query_lower = query.lower()
    
    # Load knowledge base directly for fallback
    try:
        import json
        from pathlib import Path
        kb_path = Path(__file__).parent.parent / "rag" / "knowledge_base.json"
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        # Handle pricing/plan questions
        if any(kw in query_lower for kw in ["price", "pricing", "cost", "plan", "how much"]):
            # Check if asking about specific plan
            if "pro" in query_lower:
                pro_plan = next((p for p in kb['pricing_plans'] if p['name'] == 'Pro Plan'), None)
                if pro_plan:
                    features_str = "\n".join(f"• {f}" for f in pro_plan['features'][:5])
                    return f"The Pro Plan is {pro_plan['price']}. Here are the key features:\n{features_str}\n\nThis plan is perfect for professional content creators, YouTubers, and businesses. Would you like to know more or get started?"
            
            elif "basic" in query_lower:
                basic_plan = next((p for p in kb['pricing_plans'] if p['name'] == 'Basic Plan'), None)
                if basic_plan:
                    features_str = "\n".join(f"• {f}" for f in basic_plan['features'][:5])
                    return f"The Basic Plan is {basic_plan['price']}. Here are the key features:\n{features_str}\n\nWould you like to learn more about upgrading to Pro?"
            
            # General pricing question
            plans_info = []
            for plan in kb['pricing_plans']:
                plans_info.append(f"• {plan['name']}: {plan['price']}")
            
            return f"AutoStream offers two pricing plans:\n\n" + "\n".join(plans_info) + "\n\nBoth plans include a 14-day free trial with no credit card required. The Pro Plan includes unlimited videos, 4K resolution, AI-powered features, and 24/7 support. Would you like more details about either plan?"
        
        # Default fallback
        return "I'd be happy to help! AutoStream offers two plans: Basic at $29/month and Pro at $79/month. Both include our AI-powered video editing tools. What would you like to know more about?"
    
    except Exception as e:
        print(f"⚠️ Error loading knowledge base for fallback: {e}")
        return "I'd be happy to help! AutoStream offers two plans: Basic at $29/month and Pro at $79/month. Both include our AI-powered video editing tools. What would you like to know more about?"


def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Answer user questions using RAG from knowledge base
    Optimized for speed - uses fast fallback for pricing questions
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    print(f"\n--- [DEBUG] RAG Node ---")
    print(f"User Query: {last_message}")
    
    retriever = None
    query_lower = last_message.lower()
    
    # Fast path: For pricing/plan questions, use fallback directly (no LLM call)
    pricing_keywords = ["price", "pricing", "cost", "how much", "plan", "pro plan", "basic plan", "pro price", "basic price"]
    is_pricing_query = any(keyword in query_lower for keyword in pricing_keywords)
    
    if is_pricing_query:
        print("💰 Pricing query detected - using fast fallback (skipping LLM)")
        try:
            retriever = get_retriever()
            ai_response = _get_fallback_response(last_message, retriever)
            return {
                "messages": [AIMessage(content=ai_response)],
                "next_action": "end"
            }
        except Exception as e:
            print(f"⚠️ Error in fast fallback: {e}")
            # Continue to LLM path as backup
    
    # Standard path: Use LLM with RAG for other questions
    context = None
    
    try:
        # Retrieve relevant context
        retriever = get_retriever()
        print("Retrieving context...")
        context = retriever.get_context(last_message, k=3)
        print(f"Context retrieved (first 100 chars): {context[:100]}...")
        
        # Create RAG prompt with better structure
        system_prompt = f"""You are a helpful sales assistant for AutoStream, an automated video editing SaaS platform.

IMPORTANT: Answer the user's question using ONLY the information provided in the context below. Be direct and helpful.

Context from knowledge base:
{context}

User question: {last_message}

Instructions:
1. Answer the question directly using the context above
2. Be friendly, professional, and conversational
3. If asked about pricing or plans, provide specific prices and key features
4. If the context doesn't contain the answer, say you don't have that specific information
5. Keep your response concise but informative

Now answer the user's question:"""
        
        print(f"--- [LLM PROMPT START] ---")
        print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
        print(f"--- [LLM PROMPT END] ---")
        
        print("Invoking LLM...")
        response = llm.invoke([HumanMessage(content=system_prompt)])
        
        print(f"RAW LLM RESPONSE OBJECT: {response}")
        ai_response = response.content if hasattr(response, 'content') else None
        
        # Handle empty or None response
        if not ai_response or (isinstance(ai_response, str) and len(ai_response.strip()) == 0):
            print("⚠️ LLM returned empty or None content!")
            # Check for safety filter info in response metadata
            if hasattr(response, 'response_metadata'):
                print(f"Response Metadata: {response.response_metadata}")
            # Use fallback with knowledge base
            ai_response = _get_fallback_response(last_message, retriever)
        else:
            print(f"LLM Response length: {len(ai_response)}")
            print("LLM invocation successful.")
            # Validate response is meaningful (not just error message)
            if len(ai_response.strip()) < 20:
                print("⚠️ LLM response too short, using fallback")
                ai_response = _get_fallback_response(last_message, retriever)
        
    except Exception as e:
        print(f"❌ Error in RAG Node: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Use fallback response with knowledge base data
        ai_response = _get_fallback_response(last_message, retriever)
    
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
    
    print(f"\n--- [DEBUG] Lead Capture Node ---")
    print(f"Waiting for: {waiting_for}")
    print(f"Current lead_info: {lead_info}")
    print(f"Last message: {last_message}")
    
    # If we just detected high intent, start collection
    if waiting_for is None:
        response = "That's great! I'd love to help you get started with the Pro plan. Can I get your name first?"
        print("Starting lead capture - asking for name")
        return {
            "messages": [AIMessage(content=response)],
            "lead_info": lead_info,
            "waiting_for": "name",
            "next_action": "end"
        }
    
    # Extract information based on what we're waiting for
    extracted = extract_lead_info(last_message, waiting_for)
    print(f"Extracted info: {extracted}")
    lead_info.update(extracted)
    print(f"Updated lead_info: {lead_info}")
    
    # Determine next step based on what's missing
    if not lead_info.get("name") or lead_info.get("name") == "" or lead_info.get("name") is None:
        response = "Could you please provide your name?"
        next_waiting = "name"
        print("Still waiting for name")
    elif not lead_info.get("email") or lead_info.get("email") == "" or lead_info.get("email") is None:
        response = f"Thanks, {lead_info['name']}! What's your email address?"
        next_waiting = "email"
        print(f"Got name: {lead_info['name']}, now asking for email")
    elif not lead_info.get("platform") or lead_info.get("platform") == "" or lead_info.get("platform") is None:
        response = "Great! Which platform do you primarily create content for? (e.g., YouTube, Instagram, TikTok)"
        next_waiting = "platform"
        print(f"Got email: {lead_info['email']}, now asking for platform")
    else:
        # All information collected, trigger tool
        next_waiting = None
        response = None  # Will be set by tool execution
        print("All info collected, triggering tool")
    
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
    print(f"\n--- [DEBUG] Tool Execution Node ---")
    lead_info = state.get("lead_info", {})
    
    print(f"Lead info received: {lead_info}")
    
    # Validate that we have all required fields
    name = lead_info.get("name")
    email = lead_info.get("email")
    platform = lead_info.get("platform")
    
    if not name or not email or not platform:
        error_msg = f"Missing required fields. Name: {name}, Email: {email}, Platform: {platform}"
        print(f"❌ {error_msg}")
        response = f"I'm sorry, I'm missing some information. Could you please provide:\n"
        if not name:
            response += "- Your name\n"
        if not email:
            response += "- Your email address\n"
        if not platform:
            response += "- Your content platform\n"
        response += "\nLet's try again!"
        
        return {
            "messages": [AIMessage(content=response)],
            "next_action": "end"
        }
    
    try:
        # Call the mock lead capture function
        print(f"Calling mock_lead_capture with: name={name}, email={email}, platform={platform}")
        result = mock_lead_capture(
            name=name,
            email=email,
            platform=platform
        )
        
        print(f"Tool execution result: {result}")
        
        if result == "SUCCESS":
            response = f"Perfect! I've got you all set up, {name}. You'll receive an email at {email} with instructions to activate your Pro plan. Welcome to AutoStream! 🎉"
        else:
            response = f"I'm sorry, there was an issue capturing your information: {result}. Please try again later."
        
    except Exception as e:
        print(f"❌ Error in tool execution: {str(e)}")
        import traceback
        traceback.print_exc()
        response = f"I'm sorry, there was a technical issue. Please try again later. Error: {str(e)}"
    
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
