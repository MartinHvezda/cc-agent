"""
IAM (Identity and Access Management) Agent
Specialized agent for handling identity and access related issues
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

load_dotenv()

# IAM Tools
@tool
def get_user_identity(email: str) -> Dict[str, Any]:
    """
    Retrieve user identity information by email
    
    Args:
        email: Customer's email address
    """
    # TODO: Implement actual IAM API call
    return {
        "identity_id": "usr_12345",
        "email": email,
        "account_status": "active",
        "mfa_enabled": True,
        "last_login": "2024-01-20T10:30:00Z",
        "failed_login_attempts": 0
    }

@tool  
def reset_password(identity_id: str, verification_method: str = "email") -> Dict[str, Any]:
    """
    Reset user password
    
    Args:
        identity_id: User's identity ID
        verification_method: Method for verification (email, sms, security_questions)
    """
    # TODO: Implement password reset via IAM API
    return {
        "status": "success",
        "action": "password_reset_initiated",
        "verification_method": verification_method,
        "reset_token_sent": True,
        "expires_in": "15 minutes"
    }

@tool
def block_user(identity_id: str, reason: str, duration: str = "indefinite") -> Dict[str, Any]:
    """
    Block user account
    
    Args:
        identity_id: User's identity ID  
        reason: Reason for blocking
        duration: Block duration (temporary, indefinite)
    """
    # TODO: Implement user blocking via IAM API
    return {
        "status": "success",
        "action": "user_blocked",
        "reason": reason,
        "duration": duration,
        "blocked_at": "2024-01-20T15:45:00Z"
    }

@tool
def unblock_user(identity_id: str, reason: str) -> Dict[str, Any]:
    """
    Unblock user account
    
    Args:
        identity_id: User's identity ID
        reason: Reason for unblocking
    """
    # TODO: Implement user unblocking via IAM API
    return {
        "status": "success", 
        "action": "user_unblocked",
        "reason": reason,
        "unblocked_at": "2024-01-20T16:00:00Z"
    }

@tool
def reset_mfa_devices(identity_id: str, device_type: str = "all") -> Dict[str, Any]:
    """
    Reset MFA devices for user
    
    Args:
        identity_id: User's identity ID
        device_type: Type of device to reset (all, authenticator, sms, hardware)
    """
    # TODO: Implement MFA reset via IAM API
    return {
        "status": "success",
        "action": "mfa_devices_reset",
        "device_type": device_type,
        "backup_codes_generated": True,
        "reset_at": "2024-01-20T16:15:00Z"
    }

@tool
def verify_identity(identity_id: str, verification_data: str) -> Dict[str, Any]:
    """
    Verify customer identity using provided data
    
    Args:
        identity_id: User's identity ID
        verification_data: Identity verification data (JSON string)
    """
    # TODO: Implement identity verification logic
    return {
        "verified": True,
        "verification_method": "security_questions",
        "confidence_score": 0.95,
        "verified_at": "2024-01-20T16:30:00Z"
    }

@tool
def check_account_status(identity_id: str) -> Dict[str, Any]:
    """
    Check detailed account status and security flags
    
    Args:
        identity_id: User's identity ID
    """
    # TODO: Implement account status check via IAM API
    return {
        "account_status": "active",
        "security_flags": [],
        "password_last_changed": "2024-01-01T00:00:00Z",
        "mfa_status": "enabled",
        "login_restrictions": None,
        "account_locked": False
    }

@tool
def get_login_history(identity_id: str, days: int = 7) -> Dict[str, Any]:
    """
    Get user's login history for troubleshooting
    
    Args:
        identity_id: User's identity ID
        days: Number of days to look back
    """
    # TODO: Implement login history retrieval
    return {
        "login_attempts": [
            {
                "timestamp": "2024-01-20T10:30:00Z",
                "status": "success",
                "ip_address": "192.168.1.100",
                "user_agent": "Chrome/120.0"
            },
            {
                "timestamp": "2024-01-19T09:15:00Z", 
                "status": "failed",
                "ip_address": "192.168.1.100",
                "failure_reason": "invalid_password"
            }
        ],
        "total_attempts": 2,
        "failed_attempts": 1
    }

# IAM Agent Tools List
iam_tools = [
    get_user_identity,
    reset_password,
    block_user,
    unblock_user,
    reset_mfa_devices,
    verify_identity,
    check_account_status,
    get_login_history
]

# Create IAM Agent
def create_iam_agent(llm: ChatOpenAI):
    """Create IAM agent with specialized tools and prompt"""
    
    system_prompt = """You are an IAM (Identity and Access Management) specialist agent. Your expertise includes:

CAPABILITIES:
- Password reset and recovery
- Account blocking/unblocking
- MFA (Multi-Factor Authentication) management
- Identity verification  
- Login troubleshooting
- Account security management

PROCESS:
1. Always verify customer identity first using get_user_identity
2. Check account status and security flags with check_account_status
3. Review recent login history if relevant using get_login_history
4. Apply appropriate resolution based on issue type
5. Confirm resolution with customer

SECURITY PROTOCOLS:
- Never bypass security measures
- Always verify identity before sensitive operations
- Document all actions taken
- Escalate suspicious activity immediately

Be thorough, security-conscious, and customer-focused in your approach.
Provide clear explanations of actions taken and next steps for the customer."""

    return create_react_agent(
        model=llm,
        tools=iam_tools,
        state_modifier=system_prompt
    )

# IAM Agent instance
iam_agent = None

def get_iam_agent():
    """Get or create IAM agent instance"""
    global iam_agent
    if iam_agent is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        iam_agent = create_iam_agent(llm)
    return iam_agent