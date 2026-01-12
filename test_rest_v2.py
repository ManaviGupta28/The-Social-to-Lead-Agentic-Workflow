
import os
import subprocess
from dotenv import load_dotenv
load_dotenv()

def test_curl_v2():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found!")
        return
    
    # Use a simpler curl command that prints the full response
    cmd = f"""
    $body = '{{ "contents": [{{ "parts": [{{ "text": "Hello" }}] }}] }}'
    $url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    try {{
        $response = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $body
        Write-Output "✅ REST API Success!"
        $response | ConvertTo-Json
    }} catch {{
        Write-Output "❌ REST API Failed!"
        # Explicitly read the response stream for 400 errors
        $streamReader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $errorResponse = $streamReader.ReadToEnd()
        Write-Output $errorResponse
    }}
    """
    
    result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}")

if __name__ == "__main__":
    test_curl_v2()
