
import os
import subprocess
from dotenv import load_dotenv
load_dotenv()

def list_models_rest():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found!")
        return
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    cmd = f"Invoke-RestMethod -Uri '{url}' -Method Get | ConvertTo-Json"
    
    result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}")

if __name__ == "__main__":
    list_models_rest()
