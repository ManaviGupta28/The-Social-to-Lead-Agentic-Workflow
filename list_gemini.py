
import os
import subprocess
import json
from dotenv import load_dotenv
load_dotenv()

def list_models_parsed():
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    cmd = f"Invoke-RestMethod -Uri '{url}' -Method Get | ConvertTo-Json -Depth 10"
    
    result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
    if result.stdout:
        try:
            data = json.loads(result.stdout)
            models = data.get("models", [])
            print("Found Gemini models:")
            for m in models:
                name = m.get("name")
                if "gemini" in name.lower():
                    print(f"- {name} ({m.get('displayName')})")
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(result.stdout[:500])
    
if __name__ == "__main__":
    list_models_parsed()
