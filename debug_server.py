
import sys
import traceback

def run_server():
    with open("server_log.txt", "w", buffering=1) as f:
        sys.stdout = f
        sys.stderr = f
        
        print("🚀 Starting Server with Debug Logging")
        try:
            import main
            # In case uvicorn is started in main.py, it will run here
        except Exception as e:
            print("\n❌ SERVER CRASHED")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    run_server()
