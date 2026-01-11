"""
LangGraph Workflow Definition

Defines the conversational agent graph with:
- State management using checkpointers
- Conditional routing based on intent
- Multi-turn conversation support
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    intent_classifier_node,
    rag_node,
    lead_capture_node,
    tool_execution_node,
    greeting_node
)


def route_by_intent(state: AgentState) -> str:
    """
    Route to appropriate node based on detected intent
    """
    intent = state.get("intent", "unknown")
    
    # If we're in lead capture mode, stay there
    if state.get("waiting_for"):
        return "lead_capture"
    
    # Route based on intent
    if intent == "greeting":
        return "greeting"
    elif intent == "inquiry":
        return "rag"
    elif intent == "high_intent":
        return "lead_capture"
    else:
        return "rag"  # Default to RAG for unknown


def route_next_action(state: AgentState) -> str:
    """
    Route based on next_action field
    """
    next_action = state.get("next_action", "end")
    
    if next_action == "execute_tool":
        return "tool_execution"
    elif next_action == "route":
        return "router"
    else:
        return "end"


# Build the graph
def create_agent_graph():
    """
    Create and compile the LangGraph workflow
    
    Returns:
        Compiled graph ready for execution
    """
    # Initialize graph with state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("lead_capture", lead_capture_node)
    workflow.add_node("tool_execution", tool_execution_node)
    
    # Set entry point
    workflow.set_entry_point("intent_classifier")
    
    # Add conditional edges from intent classifier
    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "greeting": "greeting",
            "rag": "rag",
            "lead_capture": "lead_capture"
        }
    )
    
    # Add edges from other nodes
    workflow.add_conditional_edges(
        "greeting",
        route_next_action,
        {
            "end": END,
            "router": "intent_classifier"
        }
    )
    
    workflow.add_conditional_edges(
        "rag",
        route_next_action,
        {
            "end": END,
            "router": "intent_classifier"
        }
    )
    
    workflow.add_conditional_edges(
        "lead_capture",
        route_next_action,
        {
            "end": END,
            "tool_execution": "tool_execution",
            "router": "intent_classifier"
        }
    )
    
    workflow.add_conditional_edges(
        "tool_execution",
        route_next_action,
        {
            "end": END,
            "router": "intent_classifier"
        }
    )
    
    # Compile with checkpointer for state persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# Create the agent instance
agent_graph = create_agent_graph()
