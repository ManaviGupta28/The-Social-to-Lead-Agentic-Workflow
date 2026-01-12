
import os
import subprocess
from dotenv import load_dotenv
load_dotenv()

def test_curl():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found!")
        return
    
    # Use PowerShell to run curl (Invoke-WebRequest)
    cmd = f"""
    $body = @{{
        contents = @(
            @{{
                parts = @(
                    @{{ text = "Hello" }}
                )
            }}
        )
    }} | ConvertTo-Json
    
    $url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    try {{
        $response = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $body
        Write-Output "✅ REST API Success!"
        $response | ConvertTo-Json
    }} catch {{
        Write-Output "❌ REST API Failed!"
        Write-Output $_.Exception.Message
        if ($_.ErrorDetails) {{ Write-Output $_.ErrorDetails.Message }}
    }}
    """
    
    # Run powershell command
    result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}")

if __name__ == "__main__":
    test_curl()
