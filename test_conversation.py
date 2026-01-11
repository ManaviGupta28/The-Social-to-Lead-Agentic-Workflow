"""
Test script for AutoStream AI Agent
Simulates a 6-turn conversation to verify all functionality
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"
THREAD_ID = "test_user_001"


def send_message(turn_num, message, expected_intent=None):
    """Send a message and print the response"""
    print(f"\n{'='*60}")
    print(f"TURN {turn_num}: User Message")
    print(f"{'='*60}")
    print(f"📤 {message}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/webhook",
            json={"message": message, "thread_id": THREAD_ID},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n📥 Agent Response:")
            print(f"   {data['response']}")
            print(f"\n🎯 Detected Intent: {data.get('intent', 'N/A')}")
            
            if expected_intent and data.get('intent') != expected_intent:
                print(f"   ⚠️  WARNING: Expected intent '{expected_intent}'")
            
            return data
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server. Is it running on port 8000?")
        return None
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None


def main():
    """Run the 6-turn conversation test"""
    print("\n" + "="*60)
    print("AutoStream AI Agent - 6-Turn Conversation Test")
    print("="*60)
    print("\nThis script tests:")
    print("✓ RAG-based knowledge retrieval")
    print("✓ Intent detection (greeting → inquiry → high-intent)")
    print("✓ State management across turns")
    print("✓ Lead information collection")
    print("✓ Tool execution (mock_lead_capture)")
    print("\n" + "="*60)
    
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("\n❌ Server health check failed!")
            return
    except:
        print("\n❌ Cannot connect to server!")
        print("Please start the server with: python main.py")
        return
    
    print("\n✅ Server is running!\n")
    time.sleep(1)
    
    # Turn 1: Pricing inquiry (RAG test)
    send_message(
        1,
        "Hi, how much is the Pro plan?",
        expected_intent="inquiry"
    )
    time.sleep(1)
    
    # Turn 2: High-intent signal
    send_message(
        2,
        "Cool, I want to sign up for my YouTube channel.",
        expected_intent="high_intent"
    )
    time.sleep(1)
    
    # Turn 3: Provide name
    send_message(
        3,
        "My name is John Doe"
    )
    time.sleep(1)
    
    # Turn 4: Provide email
    send_message(
        4,
        "john.doe@example.com"
    )
    time.sleep(1)
    
    # Turn 5: Provide platform (should trigger tool)
    print("\n" + "="*60)
    print("📌 WATCH CONSOLE: Tool execution should happen now...")
    print("="*60)
    send_message(
        5,
        "YouTube"
    )
    time.sleep(1)
    
    # Turn 6: Follow-up question
    send_message(
        6,
        "How do I cancel if I need to?"
    )
    
    # Summary
    print("\n" + "="*60)
    print("Test Completed!")
    print("="*60)
    print("\n✅ Check the console output above for:")
    print("   1. RAG response about Pro plan pricing (Turn 1)")
    print("   2. Intent shift to 'high_intent' (Turn 2)")
    print("   3. Successful collection of name, email, platform")
    print("   4. 🎯 Lead capture confirmation (should appear in server logs)")
    print("   5. Continued conversation after lead capture")
    print("\n")


if __name__ == "__main__":
    main()
