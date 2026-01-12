
print("Step 1: Importing langchain_huggingface...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Import successful")
    
    print("Step 2: Initializing embeddings model...")
    # Using a tiny model for testing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Initialization successful")
    
    print("Step 3: Testing embedding...")
    res = embeddings.embed_query("test")
    print(f"✅ Embedding successful, size: {len(res)}")
    
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
