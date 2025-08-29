#!/usr/bin/env python3
"""
Test runner script for VLAM project.
Provides convenient interface for running different test suites.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="VLAM Test Runner")
    parser.add_argument(
        "--suite", 
        choices=["all", "unit", "performance", "examples", "quick"],
        default="quick",
        help="Test suite to run (default: quick)"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU tests"
    )
    parser.add_argument(
        "--no-slow",
        action="store_true", 
        help="Skip slow tests"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])
    
    # Coverage options
    if args.coverage:
        pytest_cmd.extend([
            "--cov=vlam",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Marker exclusions
    markers = []
    if args.no_gpu:
        markers.append("not gpu")
    if args.no_slow:
        markers.append("not slow")
    
    if markers:
        pytest_cmd.extend(["-m", " and ".join(markers)])
    
    # Test suite selection
    success = True
    
    if args.suite == "all":
        # Run all tests
        cmd = pytest_cmd + ["tests/"]
        success &= run_command(cmd, "All Tests")
        
    elif args.suite == "unit":
        # Run unit tests only
        cmd = pytest_cmd + ["tests/test_main.py"]
        success &= run_command(cmd, "Unit Tests")
        
    elif args.suite == "performance":
        # Run performance tests
        cmd = pytest_cmd + ["tests/test_performance.py"]
        success &= run_command(cmd, "Performance Tests")
        
    elif args.suite == "examples":
        # Run example tests
        cmd = pytest_cmd + ["tests/test_examples.py"]
        success &= run_command(cmd, "Example Tests")
        
    elif args.suite == "quick":
        # Run quick tests (exclude slow and gpu)
        cmd = pytest_cmd + ["-m", "not slow and not gpu", "tests/"]
        success &= run_command(cmd, "Quick Tests")
    
    # Run linting if requested
    if args.suite in ["all"]:
        print(f"\n{'='*60}")
        print("Running Code Quality Checks")
        print(f"{'='*60}")
        
        # Check if black is available
        try:
            subprocess.run(["black", "--version"], check=True, capture_output=True)
            success &= run_command(
                ["black", "--check", "vlam/", "tests/"], 
                "Code Formatting Check (black)"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Black not available, skipping format check")
        
        # Check if flake8 is available
        try:
            subprocess.run(["flake8", "--version"], check=True, capture_output=True)
            success &= run_command(
                ["flake8", "vlam/", "tests/"], 
                "Code Linting (flake8)"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Flake8 not available, skipping lint check")
    
    # Final report
    print(f"\n{'='*60}")
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("💥 SOME TESTS FAILED!")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
