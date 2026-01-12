
import os
import sys

# Set cache dirs
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, "models_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir

print(f"HF_HOME set to: {cache_dir}")

try:
    print("Step 1: Importing HuggingFaceEmbeddings...")
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Done.")
    
    print("Step 2: Initializing model (all-MiniLM-L6-v2)...")
    print("This may take a while if it's downloading for the first time.")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder=cache_dir
    )
    print("Step 2: Done.")
    
    print("Step 3: Testing embedding...")
    vec = embeddings.embed_query("test")
    print(f"Success! Vector length: {len(vec)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
