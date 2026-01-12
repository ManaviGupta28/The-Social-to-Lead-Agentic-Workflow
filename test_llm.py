
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def test_llm():
    print("🚀 Testing Google Gemini LLM...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found!")
        return
    
    print(f"API Key found (starts with: {api_key[:5]}...)")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7
        )
        print("Invoking LLM...")
        response = llm.invoke([HumanMessage(content="Say 'Hello' if you can hear me.")])
        print(f"✅ LLM Response: {response.content}")
    except Exception as e:
        print(f"❌ LLM Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm()
