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
        "refund", "cancel", "trial", "know more", "more info", "more information",
        "details", "tell me more", "what else"
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
        
        # Handle "know more" / "more info" / "yes" queries - provide comprehensive details
        know_more_keywords = ["know more", "more info", "more information", "tell me more", "what else", "details"]
        # Only treat "yes" as "know more" if it's a short response (likely answering "would you like to know more?")
        if any(kw in query_lower for kw in know_more_keywords) or (query_lower.strip() == "yes" and len(query_lower.split()) == 1):
            pro_plan = next((p for p in kb['pricing_plans'] if p['name'] == 'Pro Plan'), None)
            if pro_plan:
                # Provide ALL features for "know more"
                all_features = "\n".join(f"• {f}" for f in pro_plan['features'])
                response = f"Here's everything about the Pro Plan ({pro_plan['price']}):\n\n"
                response += f"Complete Feature List:\n{all_features}\n\n"
                
                if pro_plan.get('recommended_for'):
                    response += f"Perfect For: {pro_plan['recommended_for']}\n\n"
                
                # Add relevant FAQ info
                faq_info = []
                for faq in kb.get('faq', []):
                    if any(word in faq['question'].lower() for word in ['pro', 'upgrade', 'annual', 'video length']):
                        faq_info.append(f"{faq['question']}\n{faq['answer']}")
                
                if faq_info:
                    response += "Frequently Asked Questions:\n" + "\n\n".join(faq_info[:3]) + "\n\n"
                
                # Add trial info
                trial_policy = next((p for p in kb.get('policies', []) if 'trial' in p['title'].lower()), None)
                if trial_policy:
                    response += f"{trial_policy['title']}: {trial_policy['description']}\n\n"
                
                response += "Ready to get started? Just say 'I want to get started' or 'sign me up'!"
                return response
        
        
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
    
    except Exception:
        return "I'd be happy to help! AutoStream offers two plans: Basic at $29/month and Pro at $79/month. Both include our AI-powered video editing tools. What would you like to know more about?"


def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Answer user questions using RAG from knowledge base
    Optimized for speed - uses fast fallback for pricing questions
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    retriever = None
    query_lower = last_message.lower()
    
    # Fast path: For pricing/plan/know more questions, use fallback directly (no LLM call)
    fast_path_keywords = [
        "price", "pricing", "cost", "how much", "plan", "pro plan", "basic plan", 
        "pro price", "basic price", "know more", "more info", "more information", 
        "tell me more", "what else", "details", "yes", "i want", "get started"
    ]
    is_fast_path_query = any(keyword in query_lower for keyword in fast_path_keywords)
    
    if is_fast_path_query:
        try:
            retriever = get_retriever()
            ai_response = _get_fallback_response(last_message, retriever)
            return {
                "messages": [AIMessage(content=ai_response)],
                "next_action": "end"
            }
        except Exception:
            pass  # Continue to LLM path as backup
    
    # Standard path: Use LLM with RAG for other questions
    context = None
    
    try:
        retriever = get_retriever()
        context = retriever.get_context(last_message, k=3)
        
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
        
        response = llm.invoke([HumanMessage(content=system_prompt)])
        ai_response = response.content if hasattr(response, 'content') else None
        
        if not ai_response or (isinstance(ai_response, str) and len(ai_response.strip()) == 0):
            ai_response = _get_fallback_response(last_message, retriever)
        elif len(ai_response.strip()) < 20:
            ai_response = _get_fallback_response(last_message, retriever)
        
    except Exception:
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
    
    if waiting_for is None:
        response = "That's great! I'd love to help you get started with the Pro plan. Can I get your name first?"
        return {
            "messages": [AIMessage(content=response)],
            "lead_info": lead_info,
            "waiting_for": "name",
            "next_action": "end"
        }
    
    extracted = extract_lead_info(last_message, waiting_for)
    lead_info.update(extracted)
    
    if not lead_info.get("name") or lead_info.get("name") == "" or lead_info.get("name") is None:
        response = "Could you please provide your name?"
        next_waiting = "name"
    elif not lead_info.get("email") or lead_info.get("email") == "" or lead_info.get("email") is None:
        response = f"Thanks, {lead_info['name']}! What's your email address?"
        next_waiting = "email"
    elif not lead_info.get("platform") or lead_info.get("platform") == "" or lead_info.get("platform") is None:
        response = "Great! Which platform do you primarily create content for? (e.g., YouTube, Instagram, TikTok)"
        next_waiting = "platform"
    else:
        next_waiting = None
        response = None
    
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
    lead_info = state.get("lead_info", {})
    name = lead_info.get("name")
    email = lead_info.get("email")
    platform = lead_info.get("platform")
    
    if not name or not email or not platform:
        response = "I'm sorry, I'm missing some information. Could you please provide:\n"
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
        result = mock_lead_capture(name=name, email=email, platform=platform)
        
        if result == "SUCCESS":
            response = f"Perfect! I've got you all set up, {name}. You'll receive an email at {email} with instructions to activate your Pro plan. Welcome to AutoStream! 🎉"
        else:
            response = f"I'm sorry, there was an issue capturing your information: {result}. Please try again later."
        
    except Exception as e:
        response = "I'm sorry, there was a technical issue. Please try again later."
    
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
