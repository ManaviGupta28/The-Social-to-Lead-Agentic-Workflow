
import os
import sys
import traceback

def main():
    # Use unbuffered output
    with open("diagnose_output.txt", "w", buffering=1) as f:
        sys.stdout = f
        sys.stderr = f
        
        print("🚀 Starting Diagnostic Tool (v2)")
        try:
            print(f"Current Directory: {os.getcwd()}")
            
            print("\nImporting retriever module...")
            from rag.retriever import get_retriever
            print("Import successful.")
            
            print("\nInitializing retriever instance...")
            retriever = get_retriever()
            print("Retriever initialized.")
            
            print("\nTesting context retrieval for 'Pricing'...")
            context = retriever.get_context("Pricing")
            print("Context retrieved successfully.")
            print(f"Context length: {len(context)}")
            
            print("\nTesting context retrieval for 'Pro Plan'...")
            context_pro = retriever.get_context("Pro Plan")
            print("Pro plan context retrieved.")
            print(f"First 100 chars: {context_pro[:100]}...")
            
        except Exception as e:
            print("\n❌ FAILED")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
        
        print("\n--- Diagnostic Finished ---")

if __name__ == "__main__":
    main()
