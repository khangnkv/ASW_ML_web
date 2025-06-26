#!/usr/bin/env python3
"""
Test script to verify backend connectivity
"""

import requests
import json

def test_endpoint(url, name):
    """Test a backend endpoint"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name}: Connection failed - {e}")
        return False

def main():
    print("🔍 Testing Backend Connectivity")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    
    # Test endpoints
    endpoints = [
        (f"{base_url}/api/health", "Health Check"),
        (f"{base_url}/api/models", "Models Endpoint"),
        (f"{base_url}/api/storage/stats", "Storage Stats"),
    ]
    
    all_passed = True
    for url, name in endpoints:
        if not test_endpoint(url, name):
            all_passed = False
    
    print("=" * 40)
    if all_passed:
        print("🎉 All endpoints are working!")
        print("✅ Backend is ready for frontend connection")
    else:
        print("❌ Some endpoints failed")
        print("🔧 Please check your backend configuration")
    
    # Test CORS headers
    print("\n🔍 Testing CORS Configuration...")
    try:
        response = requests.get(f"{base_url}/api/health")
        cors_header = response.headers.get('Access-Control-Allow-Origin')
        if cors_header:
            print(f"✅ CORS configured: {cors_header}")
        else:
            print("⚠️  CORS header not found")
    except Exception as e:
        print(f"❌ CORS test failed: {e}")

if __name__ == "__main__":
    main() 