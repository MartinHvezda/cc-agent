"""
Customer Care Agent System using LangGraph
Supervisor pattern with specialized agents using create_react_agent
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage

# Import our specialized agents
from agents.cc_supervisor import get_supervisor_agent
from agents.iam_agent import get_iam_agent

load_dotenv()

# State definition for the agent workflow
class CustomerCareState(TypedDict):
    issue: str
    customer_email: str
    resolution_status: str
    messages: List[Dict[str, Any]]
    escalation_needed: bool
    assigned_agent: Optional[str]
    context: Dict[str, Any]
    supervisor_analysis: Optional[str]
    agent_response: Optional[str]

# LangGraph workflow nodes
def supervisor_node(state: CustomerCareState) -> CustomerCareState:
    """Main supervisor node that investigates and routes issues"""
    print(f"🔍 Supervisor analyzing issue: {state['issue'][:50]}...")
    
    # Get supervisor agent and process issue
    supervisor = get_supervisor_agent()
    
    # Create a prompt for the supervisor
    prompt = f"""
    New customer issue received:
    
    Customer Email: {state['customer_email']}
    Issue: {state['issue']}
    
    Please investigate this issue thoroughly:
    1. Search for similar issues in our knowledge base
    2. Get customer context and history  
    3. Determine the best specialist agent to handle this
    4. Route the issue appropriately
    
    Provide your analysis and routing decision.
    """
    
    # Process with supervisor agent
    result = supervisor.invoke({
        "messages": [HumanMessage(content=prompt)]
    })
    
    # Extract the supervisor's analysis
    supervisor_response = result["messages"][-1].content
    state["supervisor_analysis"] = supervisor_response
    state["context"]["investigation_complete"] = True
    
    # Determine routing based on issue content
    issue_lower = state["issue"].lower()
    iam_keywords = ["password", "login", "access", "mfa", "2fa", "authentication", "account locked", "reset", "block"]
    
    if any(keyword in issue_lower for keyword in iam_keywords):
        state["assigned_agent"] = "IAM_Agent"
        print("📋 Routing to IAM Agent")
    else:
        state["resolution_status"] = "needs_escalation"
        print("⚠️ Issue requires human escalation")
    
    return state

def iam_agent_node(state: CustomerCareState) -> CustomerCareState:
    """IAM agent processing node"""
    print(f"🔐 IAM Agent processing issue for {state['customer_email']}")
    
    # Get IAM agent and process issue
    iam_agent = get_iam_agent()
    
    # Create a prompt for the IAM agent
    prompt = f"""
    IAM Issue Resolution Request:
    
    Customer Email: {state['customer_email']}
    Issue Description: {state['issue']}
    
    Please handle this IAM issue following security protocols:
    
    1. First, get the user's identity information
    2. Verify their identity if needed for sensitive operations
    3. Check account status and any security flags
    4. Review login history if relevant to the issue
    5. Apply the appropriate resolution (password reset, unblock, MFA reset, etc.)
    6. Confirm the resolution and provide clear next steps to the customer
    
    Always prioritize security while providing excellent customer service.
    """
    
    # Process with IAM agent
    result = iam_agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })
    
    # Extract the IAM agent's response
    iam_response = result["messages"][-1].content
    state["agent_response"] = iam_response
    state["resolution_status"] = "resolved"
    
    print("✅ IAM Agent completed processing")
    return state

def should_continue(state: CustomerCareState) -> str:
    """Determine next step in workflow"""
    
    # If resolved, end the workflow
    if state.get("resolution_status") == "resolved":
        return END
    
    # If escalation needed, end with escalation
    if state.get("resolution_status") == "needs_escalation":
        return END
    
    # If assigned to IAM agent and not yet processed
    if (state.get("assigned_agent") == "IAM_Agent" and 
        state.get("agent_response") is None):
        return "iam_agent"
    
    # Continue with supervisor
    return "supervisor"

def create_customer_care_workflow():
    """Create the main customer care workflow using LangGraph"""
    
    # Initialize the state graph
    workflow = StateGraph(CustomerCareState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("iam_agent", iam_agent_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "iam_agent": "iam_agent",
            "supervisor": "supervisor", 
            END: END
        }
    )
    
    # Add conditional edges from IAM agent
    workflow.add_conditional_edges(
        "iam_agent",
        should_continue,
        {
            "supervisor": "supervisor",
            END: END
        }
    )
    
    return workflow.compile()

def process_customer_issue(issue: str, customer_email: str):
    """Process a customer issue through the workflow"""
    
    # Create workflow
    app = create_customer_care_workflow()
    
    # Initial state
    initial_state = CustomerCareState(
        issue=issue,
        customer_email=customer_email,
        resolution_status="pending",
        messages=[],
        escalation_needed=False,
        assigned_agent=None,
        context={},
        supervisor_analysis=None,
        agent_response=None
    )
    
    print(f"\n🎯 Processing customer issue:")
    print(f"Customer: {customer_email}")
    print(f"Issue: {issue}")
    print("-" * 60)
    
    # Run the workflow
    try:
        result = app.invoke(initial_state)
        
        # Display results
        print("\n📊 RESOLUTION SUMMARY:")
        print("-" * 60)
        print(f"Status: {result['resolution_status']}")
        print(f"Assigned Agent: {result.get('assigned_agent', 'None')}")
        
        if result.get('supervisor_analysis'):
            print(f"\n🔍 Supervisor Analysis:")
            print(result['supervisor_analysis'])
        
        if result.get('agent_response'):
            print(f"\n🛠️ Agent Response:")
            print(result['agent_response'])
            
        return result
        
    except Exception as e:
        print(f"❌ Error processing issue: {str(e)}")
        return None

def main():
    """Main function to run the customer care system"""
    
    print("🚀 Customer Care Agent System with LangGraph")
    print("=" * 60)
    print("Available agents:")
    print("  - CC Supervisor (Investigation & Routing)")
    print("  - IAM Agent (Identity & Access Management)")
    print("=" * 60)
    
    # Test with sample issues
    sample_issues = [
        {
            "email": "test@email.com",
            "issue": "Hi, I have problem with my account I cannot login."
        },
        {
            "email": "user@example.com", 
            "issue": "Hi, I forgot my password. What should I do?"
        },
        {
            "email": "customer@company.com",
            "issue": "My account is locked and I can't access my MFA device."
        }
    ]
    
    # Process sample issues
    for i, sample in enumerate(sample_issues, 1):
        print(f"\n{'='*20} SAMPLE ISSUE {i} {'='*20}")
        process_customer_issue(sample["issue"], sample["email"])
        
        if i < len(sample_issues):
            print("\nContinuing to next issue...")
            print("=" * 80)

if __name__ == "__main__":
    # Check if OpenAI API key is set up
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your OpenAI API key in .env file")
        print("Copy .env.example to .env and add your API key")
        exit(1)
    
    main()