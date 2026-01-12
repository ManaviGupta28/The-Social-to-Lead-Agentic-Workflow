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
    if not name or not email or not platform:
        return "ERROR: Missing required fields"
    
    # In production, this would call an actual API endpoint
    # For now, it simulates a successful lead capture
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
        # Extract platform with fuzzy matching for common misspellings
        text_lower = text_lower.strip()
        platform_map = {
            "youtube": "YouTube",
            "youtue": "YouTube",  # Common misspelling
            "youtub": "YouTube",
            "instagram": "Instagram",
            "insta": "Instagram",
            "tiktok": "TikTok",
            "ticktok": "TikTok",
            "facebook": "Facebook",
            "fb": "Facebook",
            "twitter": "Twitter",
            "x": "Twitter",
            "twitch": "Twitch",
            "linkedin": "LinkedIn",
            "linked": "LinkedIn"
        }
        
        # Check for exact or partial matches
        for key, value in platform_map.items():
            if key in text_lower:
                result["platform"] = value
                break
        
        # If no match found, use the text as-is (capitalized)
        if not result.get("platform"):
            result["platform"] = text.strip().capitalize()
    
    return result
