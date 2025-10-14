from typing import Dict, Any
import json
import requests
import sys
import time

#!/usr/bin/env python3
"""
Test script for Figma Desktop MCP server connection
"""


# Figma MCP server configuration
FIGMA_MCP_URL = "http://127.0.0.1:3845/mcp"

def test_basic_connection():
    """Test basic HTTP connection to Figma MCP server"""
    print("Testing basic connection to Figma MCP server...")
    print(f"URL: {FIGMA_MCP_URL}")
    
    try:
        # Try a simple GET request first
        response = requests.get(FIGMA_MCP_URL, timeout=5)
        print(f"✓ Connected successfully! Status code: {response.status_code}")
        
        # Print response headers
        print("\nResponse Headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        # Try to print response content if any
        if response.text:
            print("\nResponse Content:")
            try:
                json_content = response.json()
                print(json.dumps(json_content, indent=2))
            except:
                print(response.text[:500])  # Print first 500 chars if not JSON
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Connection Error: Unable to connect to Figma Desktop MCP server")
        print("  Make sure Figma Desktop is running and the MCP server is enabled")
        return False
    except requests.exceptions.Timeout:
        print("✗ Timeout Error: Request to Figma MCP server timed out")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_mcp_protocol():
    """Test MCP protocol-specific endpoints"""
    print("\n" + "="*60)
    print("Testing MCP protocol endpoints...")
    
    # Standard MCP endpoints to test
    endpoints = [
        "/",
        "/health",
        "/status",
        "/info",
        "/capabilities",
        "/tools",
        "/resources"
    ]
    
    results = []
    
    for endpoint in endpoints:
        url = f"http://127.0.0.1:3845/mcp{endpoint}"
        print(f"\nTesting endpoint: {url}")
        
        try:
            response = requests.get(url, timeout=3)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"  Response: {json.dumps(data, indent=4)[:200]}...")
                    results.append((endpoint, True, response.status_code))
                except:
                    print(f"  Response (text): {response.text[:100]}...")
                    results.append((endpoint, True, response.status_code))
            else:
                results.append((endpoint, False, response.status_code))
                
        except requests.exceptions.RequestException as e:
            print(f"  Error: {type(e).__name__}")
            results.append((endpoint, False, None))
    
    # Summary
    print("\n" + "="*60)
    print("ENDPOINT TEST SUMMARY:")
    print("-"*60)
    
    for endpoint, success, status in results:
        status_str = f"Status {status}" if status else "No connection"
        icon = "✓" if success and status == 200 else "✗"
        print(f"{icon} {endpoint:<20} {status_str}")
    
    return results

def test_mcp_jsonrpc():
    """Test JSON-RPC protocol which MCP uses"""
    print("\n" + "="*60)
    print("Testing MCP JSON-RPC protocol...")
    
    # MCP uses JSON-RPC 2.0
    test_requests = [
        {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {}
            },
            "id": 1
        },
        {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        },
        {
            "jsonrpc": "2.0",
            "method": "resources/list",
            "params": {},
            "id": 3
        }
    ]
    
    headers = {
        "Content-Type": "application/json"
    }
    
    for req in test_requests:
        print(f"\nTesting method: {req['method']}")
        print(f"Request: {json.dumps(req, indent=2)}")
        
        try:
            response = requests.post(
                FIGMA_MCP_URL,
                json=req,
                headers=headers,
                timeout=5
            )
            
            print(f"Status: {response.status_code}")
            
            if response.text:
                try:
                    data = response.json()
                    print(f"Response: {json.dumps(data, indent=2)[:300]}...")
                except:
                    print(f"Response (raw): {response.text[:200]}...")
                    
        except Exception as e:
            print(f"Error: {e}")

def check_figma_process():
    """Check if Figma Desktop is running"""
    print("\n" + "="*60)
    print("Checking for Figma Desktop process...")
    
    import subprocess
    
    try:
        # Check for Figma process on macOS
        result = subprocess.run(
            ["pgrep", "-l", "Figma"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Figma Desktop is running:")
            print(result.stdout)
            return True
        else:
            # Try alternative check
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            if "Figma" in result.stdout:
                print("✓ Figma Desktop process found")
                return True
            else:
                print("✗ Figma Desktop does not appear to be running")
                return False
                
    except Exception as e:
        print(f"Could not check process status: {e}")
        return None

def main():
    """Main test execution"""
    print("="*60)
    print("FIGMA MCP SERVER CONNECTION TEST")
    print("="*60)
    
    # Check if Figma is running
    figma_running = check_figma_process()
    
    if figma_running is False:
        print("\n⚠️  WARNING: Figma Desktop doesn't appear to be running.")
        print("Please start Figma Desktop and ensure the MCP server is enabled.")
        print("\nContinuing with connection tests anyway...")
    
    # Test basic connection
    connection_ok = test_basic_connection()
    
    if connection_ok:
        # Test MCP endpoints
        test_mcp_protocol()
        
        # Test JSON-RPC
        test_mcp_jsonrpc()
        
        print("\n" + "="*60)
        print("✓ Connection tests completed successfully!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Connection tests failed!")
        print("\nTroubleshooting steps:")
        print("1. Ensure Figma Desktop is running")
        print("2. Check if MCP server is enabled in Figma settings")
        print("3. Verify the port 3845 is not blocked")
        print("4. Check Figma Desktop preferences for API/Plugin settings")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())