"""
State Schema for AutoStream Conversational Agent

Defines the structure of conversation state that persists across turns.
"""

from typing import TypedDict, List, Dict, Optional, Literal
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State schema for the conversational agent
    
    Attributes:
        messages: Full conversation history (list of HumanMessage, AIMessage)
        intent: Current detected user intent
        lead_info: Dictionary storing collected lead information
        next_action: Controls routing in the graph
        waiting_for: What information we're currently waiting for from user
    """
    messages: List[BaseMessage]
    intent: Literal["greeting", "inquiry", "high_intent", "unknown"]
    lead_info: Dict[str, Optional[str]]  # {name, email, platform}
    next_action: str
    waiting_for: Optional[str]  # "name", "email", "platform", or None
