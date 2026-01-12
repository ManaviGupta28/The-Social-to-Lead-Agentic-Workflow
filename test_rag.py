
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent.nodes import rag_node
from agent.state import AgentState

def test_rag_node():
    print("🚀 Starting RAG Node Isolation Test")
    
    # Mock state
    state = {
        "messages": [HumanMessage(content="Hi, how much is the Pro plan?")],
        "intent": "inquiry",
        "lead_info": {"name": None, "email": None, "platform": None},
        "next_action": "route",
        "waiting_for": None
    }
    
    print("Invoking rag_node...")
    try:
        result = rag_node(state)
        print("\n✅ Node Execution Result:")
        print(f"Response: {result['messages'][0].content}")
    except Exception as e:
        print(f"\n❌ Node Execution Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_node()
