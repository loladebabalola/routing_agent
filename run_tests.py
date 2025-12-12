#!/usr/bin/env python3
"""
Test runner for the Routing Agent Framework.

This script runs all unit and integration tests.
"""

import subprocess
import sys
import os


def run_tests(test_type="all"):
    """Run tests based on specified type."""
    
    # Change to the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Determine which tests to run
    if test_type == "unit":
        test_paths = ["tests/unit"]
        print("🧪 Running unit tests...")
    elif test_type == "integration":
        test_paths = ["tests/integration"]
        print("🔗 Running integration tests...")
    else:  # all
        test_paths = ["tests/unit", "tests/integration"]
        print("🚀 Running all tests...")
    
    # Run pytest for each test path
    failed = False
    for test_path in test_paths:
        print(f"\n📁 Running tests in {test_path}")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                test_path,
                "-v",
                "--tb=short",
                "--color=yes"
            ], capture_output=False)
            
            if result.returncode != 0:
                failed = True
                print(f"❌ Tests in {test_path} failed!")
            else:
                print(f"✅ Tests in {test_path} passed!")
                
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            failed = True
    
    return not failed


def main():
    """Main entry point."""
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Please install it first:")
        print("pip install pytest")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type not in ["unit", "integration", "all"]:
            print("❌ Invalid test type. Use: unit, integration, or all")
            sys.exit(1)
    else:
        test_type = "all"
    
    # Run tests
    success = run_tests(test_type)
    
    # Exit with appropriate code
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()