"""
Customer Care Supervisor Agent
Uses create_react_agent and vector search for issue investigation
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
from langgraph_supervisor import create_supervisor

from agents.iam_agent import iam_agent

load_dotenv()

# Supervisor Tools
@tool
def investigate_similar_issues(query: str) -> Dict[str, Any]:
    """
    Search vector database for similar customer issues and their resolutions
    
    Args:
        query: The customer issue description to search for
    """
    # TODO: Implement vector database search
    # This would typically involve:
    # 1. Embedding the query
    # 2. Searching vector database for similar issues  
    # 3. Returning relevant historical cases and solutions
    
    return {
        "similar_issues": [
            {
                "issue": "Customer cannot login - password reset needed",
                "resolution": "IAM agent resolved via password reset",
                "similarity_score": 0.85
            },
            {
                "issue": "Account locked after multiple failed attempts", 
                "resolution": "IAM agent unlocked account and reset MFA",
                "similarity_score": 0.72
            }
        ],
        "recommended_agent": "IAM_Agent",
        "confidence": 0.8
    }

@tool
def route_to_specialist_agent(agent_name: str, issue_summary: str, customer_email: str) -> Dict[str, Any]:
    """
    Route the issue to a specialist agent
    
    Args:
        agent_name: Name of the specialist agent (IAM_Agent, CRM_Agent, etc.)
        issue_summary: Brief summary of the issue
        customer_email: Customer's email address
    """
    return {
        "status": "routed",
        "assigned_agent": agent_name,
        "issue_summary": issue_summary,
        "customer_email": customer_email,
        "routing_reason": f"Issue classified as {agent_name} domain based on content analysis"
    }

@tool
def escalate_to_human(reason: str, urgency: str = "normal") -> Dict[str, Any]:
    """
    Escalate issue to human agent
    
    Args:
        reason: Reason for escalation
        urgency: Urgency level (low, normal, high, critical)
    """
    return {
        "status": "escalated",
        "escalation_reason": reason,
        "urgency": urgency,
        "human_agent_required": True
    }

@tool
def get_customer_context(customer_email: str) -> Dict[str, Any]:
    """
    Retrieve customer context and history
    
    Args:
        customer_email: Customer's email address
    """
    # TODO: Implement customer context retrieval
    # This would fetch customer profile, previous issues, account status, etc.
    
    return {
        "customer_id": "12345",
        "account_status": "active",
        "previous_issues": 2,
        "last_contact": "2024-01-15",
        "tier": "premium",
        "account_age_days": 365
    }

# Supervisor Tools List
supervisor_tools = [
    investigate_similar_issues,
    route_to_specialist_agent,
    escalate_to_human,
    get_customer_context
]

# Create Supervisor Agent
def create_supervisor_agent(llm: ChatOpenAI):
    """Create supervisor agent with investigation and routing tools"""
    
    system_prompt = """You are a Customer Care Supervisor Agent. Your role is to:

1. INVESTIGATE: First, search for similar issues in the knowledge base and logs using investigate_similar_issues
2. ANALYZE: Understand the customer's problem and context using get_customer_context
3. ROUTE: Determine which specialist agent can best handle this issue using route_to_specialist_agent
4. SUPERVISE: Monitor the resolution process and escalate if needed using escalate_to_human

Available specialist agents:
- IAM_Agent: Identity and access management issues (login, password, MFA, account blocking)
- CRM_Agent: Customer relationship and account issues
- Payments_Agent: Payment and billing related issues
- Marketing_Agent: Promotional and marketing inquiries

PROCESS:
Always start by investigating similar issues before routing.
Be thorough in your analysis and provide clear reasoning for your decisions.
Use the tools available to gather information and make informed routing decisions."""

    return create_supervisor(
        model=llm,
        tools=supervisor_tools,
        state_modifier=system_prompt,
        agents=[iam_agent]
    )

# Supervisor Agent instance
supervisor_agent = None

def get_supervisor_agent():
    """Get or create supervisor agent instance"""
    global supervisor_agent
    if supervisor_agent is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        supervisor_agent = create_supervisor_agent(llm)
    return supervisor_agent