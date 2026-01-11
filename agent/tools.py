"""
Tools for AutoStream Agent

Contains the lead capture function that gets triggered
when all user information has been collected.
"""

from typing import Dict


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock lead capture function - simulates backend API call
    
    Args:
        name: User's full name
        email: User's email address
        platform: Content creation platform (YouTube, Instagram, etc.)
        
    Returns:
        Success message
    """
    # Validation
    if not name or not email or not platform:
        return "ERROR: Missing required fields"
    
    # Simulate API call
    print(f"🎯 Lead captured successfully: {name}, {email}, {platform}")
    print(f"   Platform: {platform}")
    print(f"   Email: {email}")
    print(f"   Name: {name}")
    
    return "SUCCESS"


def validate_email(email: str) -> bool:
    """
    Simple email validation
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email appears valid
    """
    return "@" in email and "." in email.split("@")[-1]


def extract_lead_info(text: str, waiting_for: str) -> Dict[str, str]:
    """
    Extract lead information from user message
    
    Args:
        text: User message
        waiting_for: What field we're expecting (name, email, platform)
        
    Returns:
        Dictionary with extracted information
    """
    result = {}
    text_lower = text.lower()
    
    if waiting_for == "name":
        # Extract name - simple heuristic
        # Could be "My name is X", "I'm X", or just "X"
        if "name is" in text_lower:
            result["name"] = text.split("name is")[-1].strip()
        elif "i'm " in text_lower or "i am " in text_lower:
            result["name"] = text_lower.replace("i'm", "").replace("i am", "").strip()
        else:
            # Assume the whole message is the name
            result["name"] = text.strip()
    
    elif waiting_for == "email":
        # Extract email
        words = text.split()
        for word in words:
            if "@" in word:
                result["email"] = word.strip()
                break
        if not result.get("email"):
            result["email"] = text.strip()
    
    elif waiting_for == "platform":
        # Extract platform
        platforms = ["youtube", "instagram", "tiktok", "facebook", "twitter", "twitch", "linkedin"]
        for platform in platforms:
            if platform in text_lower:
                result["platform"] = platform.capitalize()
                break
        if not result.get("platform"):
            result["platform"] = text.strip()
    
    return result
