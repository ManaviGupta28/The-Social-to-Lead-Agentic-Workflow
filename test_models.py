
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def test_model(name):
    print(f"\n--- Testing model: {name} ---")
    try:
        llm = ChatGoogleGenerativeAI(
            model=name,
            temperature=0.7
        )
        response = llm.invoke([HumanMessage(content="Hello")])
        print(f"✅ Success: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return False

if __name__ == "__main__":
    models_to_try = [
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-pro"
    ]
    
    for model in models_to_try:
        if test_model(model):
            print(f"\n🎯 FOUND WORKING MODEL: {model}")
            # break # Try all to see what's best
