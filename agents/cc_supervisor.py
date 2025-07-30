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
    try:
        from services.vector_service import get_vector_service
        
        vector_service = get_vector_service()
        
        # Get queue analysis for similar issues
        queue_analysis = vector_service.get_queue_analysis(query)
        
        return {
            "similar_issues": [
                {
                    "key": issue["key"],
                    "title": issue["title"],
                    "description": issue["description"],
                    "status": issue["status"],
                    "resolved_by_queue": issue["resolved_by_queue"],
                    "comments": issue["comments"],
                    "similarity_score": issue["similarity_score"]
                }
                for issue in queue_analysis["similar_issues"]
            ],
            "queue_analysis": queue_analysis["queue_analysis"],
            "analysis_confidence": queue_analysis["analysis_confidence"],
            "search_performed": True,
            "total_results": queue_analysis["total_similar_issues"]
        }
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        # Fallback to basic analysis
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["password", "login", "access", "mfa", "2fa", "authentication"]):
            return {
                "similar_issues": [
                    {
                        "key": "CS-FALLBACK",
                        "title": "Cannot Login - Password Reset",
                        "description": "Customer cannot login - password reset needed",
                        "status": "Done",
                        "resolved_by_queue": "IAM",
                        "comments": ["Password reset resolved the issue"],
                        "similarity_score": 0.75
                    }
                ],
                "queue_analysis": {"IAM": {"count": 1, "percentage": 100.0, "avg_similarity": 0.75}},
                "analysis_confidence": 0.7,
                "search_performed": False,
                "fallback_used": True
            }
        else:
            return {
                "similar_issues": [],
                "queue_analysis": {},
                "analysis_confidence": 0.0,
                "search_performed": False,
                "fallback_used": True
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

1. INVESTIGATE: Search for similar issues using investigate_similar_issues to understand how similar problems were resolved
2. ANALYZE: Review the queue analysis to see which teams/queues have successfully handled similar issues  
3. DECIDE: Make intelligent routing decisions based on historical patterns and issue characteristics
4. ROUTE: Direct issues to appropriate AI agents or escalate to human teams when needed

IMPORTANT CONTEXT:
- Most historical issues were resolved by human teams/queues (IAM, CRM, Payments, Marketing, etc.)
- AI agents (IAM_Agent) are new and should only handle issues they're specifically designed for
- Use historical queue patterns to inform your routing decisions
- The queue_analysis shows which human teams have successfully resolved similar issues

ROUTING LOGIC:
- For IAM issues (login, password, MFA): Route to IAM_Agent if it can handle it, otherwise escalate to IAM queue
- For other issues: Analyze historical queue patterns and route accordingly  
- Always explain your routing decision based on the evidence from similar issues

PROCESS:
1. Investigate similar issues and analyze queue patterns
2. Get customer context if needed
3. Make routing decision based on evidence and agent capabilities
4. Provide clear reasoning for your decision"""

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