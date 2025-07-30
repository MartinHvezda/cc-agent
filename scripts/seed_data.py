"""
Seed script for populating vector database with sample customer support issues
This is separate from production code to avoid accidentally seeding in production
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_service import get_vector_service

def seed_sample_issues():
    """Seed the database with sample customer issues in Jira format"""
    sample_issues = [
        {
            "key": "IAM-1001",
            "title": "Cannot Login - Forgot Password",
            "description": "I can't log into my account, I think I forgot my password. Getting error message when trying to sign in.",
            "status": "Done",
            "resolved_by_queue": "IAM",
            "comments": [
                "Human agent: Password reset email sent to customer",
                "Human agent: Customer confirmed successful login after reset",
                "Resolved by: Sarah from IAM team"
            ]
        },
        {
            "key": "IAM-1002",
            "title": "Account Locked - Multiple Failed Attempts",
            "description": "My account is locked after multiple failed login attempts. Cannot access my dashboard or any services.",
            "status": "Done",
            "resolved_by_queue": "IAM",
            "comments": [
                "Human agent: Account unlocked by IAM team",
                "Human agent: MFA device reset performed",
                "Human agent: Customer verified identity successfully",
                "Resolved by: Mike from IAM team"
            ]
        },
        {
            "key": "IAM-1003",
            "title": "Lost 2FA Device Access",
            "description": "I lost access to my 2FA device and cannot login. Need help resetting multi-factor authentication.",
            "status": "Done", 
            "resolved_by_queue": "IAM",
            "comments": [
                "Human agent: Identity verification completed via security questions",
                "Human agent: MFA device reset and new backup codes generated",
                "Resolved by: Lisa from IAM team"
            ]
        },
        {
            "key": "IAM-1004",
            "title": "Authentication Errors",
            "description": "Cannot access my account, getting authentication errors every time I try to login.",
            "status": "Done",
            "resolved_by_queue": "IAM", 
            "comments": [
                "Human agent: Identity verified through alternate method",
                "Human agent: Password reset completed successfully",
                "Resolved by: Tom from IAM team"
            ]
        },
        {
            "key": "CRM-1005",
            "title": "Credit Card Not Showing",
            "description": "My credit card is not showing up in my account payment methods. It was there yesterday.",
            "status": "Done",
            "resolved_by_queue": "CRM",
            "comments": [
                "Human agent: Payment method sync issue identified",
                "Human agent: Account refresh resolved the display issue",
                "Resolved by: Amanda from CRM team"
            ]
        },
        {
            "key": "PMT-1006",
            "title": "Duplicate Charge",
            "description": "I was charged twice for the same transaction on my latest invoice. Need refund for duplicate charge.",
            "status": "Done",
            "resolved_by_queue": "PMT",
            "comments": [
                "Human agent: Duplicate charge confirmed in billing system",
                "Human agent: Refund processed within 3-5 business days",
                "Resolved by: Carlos from Payments team"
            ]
        },
        {
            "key": "CRM-1007",
            "title": "Cancel Subscription",
            "description": "How do I cancel my subscription? Cannot find the cancellation option in my account settings.",
            "status": "Done",
            "resolved_by_queue": "CRM",
            "comments": [
                "Human agent: Cancellation process explained to customer",
                "Human agent: Subscription cancelled per customer request",
                "Resolved by: Jennifer from CRM team"
            ]
        },
        {
            "key": "IAM-1008",
            "title": "Update Email Address",
            "description": "I need to update my email address associated with my account. Current email is no longer active.",
            "status": "Done",
            "resolved_by_queue": "IAM",
            "comments": [
                "Human agent: Identity verification completed",
                "Human agent: Email address updated successfully",
                "Resolved by: David from IAM team"
            ]
        },
        {
            "key": "CRM-1009",
            "title": "Billing Address Update",
            "description": "Need to update my billing address for my subscription. Moved to a new location last month.",
            "status": "Done",
            "resolved_by_queue": "CRM",
            "comments": [
                "Human agent: Billing address updated in customer profile",
                "Human agent: Changes will reflect in next billing cycle",
                "Resolved by: Maria from CRM team"
            ]
        },
        {
            "key": "PMT-1010",
            "title": "Payment Method Declined",
            "description": "My payment method was declined but the card is valid. Need help updating payment information.",
            "status": "Done",
            "resolved_by_queue": "PMT",
            "comments": [
                "Human agent: Card verification completed successfully",
                "Human agent: Payment method updated and retry successful",
                "Resolved by: Robert from Payments team"
            ]
        }
    ]
    
    print("🌱 Seeding sample issues...")
    vector_service = get_vector_service()
    
    for issue_data in sample_issues:
        try:
            vector_service.add_issue(
                title=issue_data["title"],
                description=issue_data["description"],
                key=issue_data["key"],
                status=issue_data["status"],
                resolved_by_queue=issue_data["resolved_by_queue"],
                comments=issue_data["comments"]
            )
        except Exception as e:
            print(f"⚠️ Error seeding issue {issue_data['key']}: {e}")
    
    print(f"✅ Seeded {len(sample_issues)} sample issues")

def main():
    """Main function to run the seeding script"""
    print("🚀 Customer Support Issue Seeding Script")
    print("=" * 50)
    
    # Confirm before seeding
    response = input("Are you sure you want to seed the database with sample issues? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        seed_sample_issues()
        print("\n✅ Seeding completed!")
    else:
        print("❌ Seeding cancelled.")

if __name__ == "__main__":
    main()