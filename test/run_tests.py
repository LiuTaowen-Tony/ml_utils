#!/usr/bin/env python3
"""
Test runner for checkpointer tests.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test(test_file, description):
    """Run a single test file"""
    print(f"\n{'='*50}")
    print(f"🧪 {description}")
    print(f"{'='*50}")
    
    test_path = Path(__file__).parent / test_file
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    cmd = ["python", str(test_path)]
    
    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("Error:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def main():
    print("🎯 Checkpointer Test Suite")
    
    # Check GPU availability
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
    except:
        gpu_count = 0
        print("CUDA not available")
    
    results = []
    
    # Core tests that work reliably
    tests = [
        ("test_checkpointer_standalone.py", "Core FSDP1 Tests"),
        ("test_2_device_simple.py", "2-Device FSDP Tests"),
    ]
    
    # Run tests
    for test_file, description in tests:
        success = run_test(test_file, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description}: {status}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("💥 SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())