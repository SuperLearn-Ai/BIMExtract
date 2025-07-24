#!/usr/bin/env python3
"""
Test script for deployed Ultra-Cost-Optimized Pipeline on RunPod
Tests both Qwen3 LLM and Visual Parsing services
"""

import requests
import json
import base64
import time
from pathlib import Path

# Configuration
RUNPOD_ENDPOINT = "https://jeqrwyd0hbl40c-8000.proxy.runpod.net"
RUNSYNC_URL = f"{RUNPOD_ENDPOINT}/runsync"

def test_qwen3_service():
    """Test Qwen3 LLM service"""
    print("🤖 Testing Qwen3 LLM Service...")
    
    test_payload = {
        "input": {
            "service": "qwen3_llm",
            "text": "Analyze this academic document chunk: Machine learning optimization techniques have shown significant improvements in computational efficiency. The key factors include gradient descent algorithms, regularization methods, and hyperparameter tuning.",
            "thinking_mode": True,
            "max_tokens": 1024,
            "temperature": 0.1
        }
    }
    
    try:
        print(f"📡 Sending request to: {RUNSYNC_URL}")
        response = requests.post(RUNSYNC_URL, json=test_payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Qwen3 LLM Test Successful!")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            print(f"   Thinking Mode: {result.get('thinking_mode', False)}")
            print(f"   Token Count: {result.get('token_count', 0)}")
            return True
        else:
            print(f"❌ Qwen3 Test Failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Qwen3 Test Error: {e}")
        return False

def test_visual_parsing_service():
    """Test Visual Parsing service with a simple test image"""
    print("\n👁️ Testing Visual Parsing Service...")
    
    # Create a simple test image (1x1 white pixel as base64)
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    test_payload = {
        "input": {
            "service": "visual_parsing",
            "image": f"data:image/png;base64,{test_image_b64}",
            "type": "ocr"
        }
    }
    
    try:
        print(f"📡 Sending request to: {RUNSYNC_URL}")
        response = requests.post(RUNSYNC_URL, json=test_payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Visual Parsing Test Successful!")
            print(f"   Results: {len(result.get('results', []))} text elements found")
            print(f"   Service: {result.get('service', 'unknown')}")
            return True
        else:
            print(f"❌ Visual Parsing Test Failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Visual Parsing Test Error: {e}")
        return False

def test_health_check():
    """Test service health"""
    print("\n🏥 Testing Service Health...")
    
    try:
        # Try to get basic response
        response = requests.get(RUNPOD_ENDPOINT, timeout=10)
        print(f"✅ Endpoint accessible: HTTP {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Ultra-Cost-Optimized Pipeline Deployment")
    print("="*60)
    print(f"🎯 Endpoint: {RUNPOD_ENDPOINT}")
    print(f"🤖 Services: Qwen3 LLM + Visual Parsing")
    print(f"💰 Target: $0.15 per 1K pages")
    print()
    
    # Run tests
    health_ok = test_health_check()
    qwen3_ok = test_qwen3_service() if health_ok else False
    visual_ok = test_visual_parsing_service() if health_ok else False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"🏥 Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"🤖 Qwen3 LLM: {'✅ PASS' if qwen3_ok else '❌ FAIL'}")
    print(f"👁️ Visual Parsing: {'✅ PASS' if visual_ok else '❌ FAIL'}")
    
    if all([health_ok, qwen3_ok, visual_ok]):
        print("\n🎉 All tests passed! Pipeline is ready for production.")
        print(f"🔗 Use endpoint: {RUNPOD_ENDPOINT}")
        print("💡 Update your run.py with --mode runpod to use this deployment")
    else:
        print("\n⚠️ Some tests failed. Check the setup and try again.")
        print("🔧 You may need to wait for models to finish downloading.")
    
    return all([health_ok, qwen3_ok, visual_ok])

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 