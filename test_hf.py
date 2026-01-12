
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

def test_embeddings():
    print("Testing HuggingFaceEmbeddings initialization...")
    start = time.time()
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ Initialization successful in {time.time() - start:.2f}s")
        
        print("Testing a simple embedding...")
        vec = embeddings.embed_query("This is a test")
        print(f"✅ Embedding successful. Vector length: {len(vec)}")
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")

if __name__ == "__main__":
    test_embeddings()
