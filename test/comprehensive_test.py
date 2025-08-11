"""
Comprehensive Customer Care Agent Test Suite
Loads issues from resources and generates detailed test reports
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from main import process_customer_issue

class TestResult:
    def __init__(self, issue_data: Dict[str, Any]):
        self.email = issue_data["email"]
        self.issue = issue_data["issue"] 
        self.category = issue_data["category"]
        self.expected_resolution = issue_data["expected_resolution"]
        self.expected_agent = issue_data["expected_agent"]
        
        # Results to be populated
        self.actual_resolution: Optional[str] = None
        self.actual_agent: Optional[str] = None
        self.supervisor_analysis: Optional[str] = None
        self.agent_response: Optional[str] = None
        self.processing_time: float = 0.0
        self.success: bool = False
        self.error: Optional[str] = None

class CustomerCareTestSuite:
    def __init__(self, test_data_file: str = "resources/test-issues.json"):
        self.test_data_file = test_data_file
        self.test_results: List[TestResult] = []
        
    def load_test_issues(self) -> List[Dict[str, Any]]:
        """Load test issues from JSON file"""
        try:
            with open(self.test_data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Test data file not found: {self.test_data_file}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing test data: {e}")
            return []
    
    def run_single_test(self, issue_data: Dict[str, Any]) -> TestResult:
        """Run a single test case and collect results"""
        test_result = TestResult(issue_data)
        
        print(f"\n🧪 Testing: {test_result.category.upper()} Issue")
        print(f"   Email: {test_result.email}")
        print(f"   Issue: {test_result.issue[:60]}...")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            # Process the issue through the main workflow
            result = process_customer_issue(test_result.issue, test_result.email)
            
            if result:
                test_result.actual_resolution = result.get("resolution_status")
                test_result.actual_agent = result.get("assigned_agent")
                test_result.supervisor_analysis = result.get("supervisor_analysis")
                test_result.agent_response = result.get("agent_response")
                
                # Determine if test passed
                resolution_match = test_result.actual_resolution == test_result.expected_resolution
                agent_match = test_result.actual_agent == test_result.expected_agent
                test_result.success = resolution_match and agent_match
                
            else:
                test_result.error = "Workflow returned None"
                
        except Exception as e:
            test_result.error = str(e)
            
        test_result.processing_time = time.time() - start_time
        
        # Print immediate results
        status_icon = "✅" if test_result.success else "❌"
        print(f"{status_icon} Expected: {test_result.expected_resolution} / {test_result.expected_agent}")
        print(f"   Actual: {test_result.actual_resolution} / {test_result.actual_agent}")
        print(f"   Time: {test_result.processing_time:.2f}s")
        
        return test_result
    
    def run_all_tests(self) -> None:
        """Run all test cases"""
        print("🚀 COMPREHENSIVE CUSTOMER CARE AGENT TEST SUITE")
        print("=" * 80)
        
        issues = self.load_test_issues()
        if not issues:
            print("❌ No test issues found!")
            return
            
        print(f"📋 Running {len(issues)} test cases...")
        print("=" * 80)
        
        for i, issue_data in enumerate(issues, 1):
            print(f"\n{'='*15} TEST CASE {i}/{len(issues)} {'='*15}")
            test_result = self.run_single_test(issue_data)
            self.test_results.append(test_result)
            
            if i < len(issues):
                print("\n" + "-" * 80)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        avg_processing_time = sum(r.processing_time for r in self.test_results) / total_tests
        
        # Group results by category
        category_stats = {}
        for result in self.test_results:
            cat = result.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0, "failed": 0}
            
            category_stats[cat]["total"] += 1
            if result.success:
                category_stats[cat]["passed"] += 1
            else:
                category_stats[cat]["failed"] += 1
        
        # Identify resolution patterns
        resolution_stats = {}
        for result in self.test_results:
            res = result.actual_resolution
            if res not in resolution_stats:
                resolution_stats[res] = 0
            resolution_stats[res] += 1
        
        # Identify agent usage
        agent_stats = {}
        for result in self.test_results:
            agent = result.actual_agent or "None"
            if agent not in agent_stats:
                agent_stats[agent] = 0
            agent_stats[agent] += 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests, 
                "failed": failed_tests,
                "pass_rate": round(pass_rate, 2),
                "avg_processing_time": round(avg_processing_time, 2)
            },
            "category_breakdown": category_stats,
            "resolution_patterns": resolution_stats,
            "agent_usage": agent_stats,
            "failed_tests": [
                {
                    "email": r.email,
                    "issue": r.issue[:100] + "..." if len(r.issue) > 100 else r.issue,
                    "category": r.category,
                    "expected": f"{r.expected_resolution}/{r.expected_agent}",
                    "actual": f"{r.actual_resolution}/{r.actual_agent}",
                    "error": r.error
                }
                for r in self.test_results if not r.success
            ]
        }
    
    def print_report(self) -> None:
        """Print formatted test report"""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("📊 TEST REPORT")
        print("=" * 80)
        
        # Summary
        summary = report["summary"]
        print(f"🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']} ✅")
        print(f"   Failed: {summary['failed']} ❌") 
        print(f"   Pass Rate: {summary['pass_rate']}%")
        print(f"   Avg Processing Time: {summary['avg_processing_time']}s")
        
        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, stats in report["category_breakdown"].items():
            pass_rate = (stats['passed'] / stats['total']) * 100
            print(f"   {category.upper()}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")
        
        # Resolution patterns
        print(f"\n🔄 RESOLUTION PATTERNS:")
        for resolution, count in report["resolution_patterns"].items():
            print(f"   {resolution}: {count}")
        
        # Agent usage
        print(f"\n🤖 AGENT USAGE:")
        for agent, count in report["agent_usage"].items():
            print(f"   {agent}: {count}")
        
        # Failed tests
        if report["failed_tests"]:
            print(f"\n❌ FAILED TESTS ({len(report['failed_tests'])}):")
            for i, failure in enumerate(report["failed_tests"], 1):
                print(f"   {i}. [{failure['category'].upper()}] {failure['email']}")
                print(f"      Issue: {failure['issue']}")
                print(f"      Expected: {failure['expected']}")
                print(f"      Actual: {failure['actual']}")
                if failure['error']:
                    print(f"      Error: {failure['error']}")
                print()
        
        print("=" * 80)
    
    def save_report_to_file(self, filename: Optional[str] = None) -> str:
        """Save detailed report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "report": self.generate_report(),
            "detailed_results": [
                {
                    "email": r.email,
                    "issue": r.issue,
                    "category": r.category,
                    "expected_resolution": r.expected_resolution,
                    "expected_agent": r.expected_agent,
                    "actual_resolution": r.actual_resolution,
                    "actual_agent": r.actual_agent,
                    "processing_time": r.processing_time,
                    "success": r.success,
                    "error": r.error,
                    "supervisor_analysis": r.supervisor_analysis,
                    "agent_response": r.agent_response
                }
                for r in self.test_results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return filename

def main():
    """Main test execution"""
    # Check if OpenAI API key is set up
    if not os.getenv("OPENAI_API_KEY"):
        return
    
    # Run comprehensive tests
    test_suite = CustomerCareTestSuite()
    test_suite.run_all_tests()
    
    # Generate and display report
    test_suite.print_report()
    
    # Save detailed report
    report_file = test_suite.save_report_to_file()
    print(f"📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()