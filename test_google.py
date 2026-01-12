
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

def test_google_embeddings():
    print("Testing GoogleGenerativeAIEmbeddings...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("✅ Initialization successful")
        
        print("Testing a small query...")
        res = embeddings.embed_query("Pricing")
        print(f"✅ Embedding successful, size: {len(res)}")
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

if __name__ == "__main__":
    test_google_embeddings()
