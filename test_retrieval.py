
from rag.retriever import get_retriever
import time

def test_retrieval():
    print("🚀 Initializing retriever...")
    start = time.time()
    try:
        retriever = get_retriever()
        print(f"✅ Retriever initialized in {time.time() - start:.2f}s")
        
        query = "What is the Pro plan?"
        print(f"🔍 Testing retrieval for: '{query}'")
        start = time.time()
        context = retriever.get_context(query)
        print(f"✅ Retrieval successful in {time.time() - start:.2f}s")
        print("Context found:")
        print("-" * 50)
        print(context[:500] + "...")
        print("-" * 50)
    except Exception as e:
        print(f"❌ Retrieval failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()
