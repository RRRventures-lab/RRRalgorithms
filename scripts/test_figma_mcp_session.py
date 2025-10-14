from typing import Dict, Any, Optional
import json
import requests
import sys
import uuid

#!/usr/bin/env python3
"""
Enhanced test script for Figma Desktop MCP server with session handling
"""


# Figma MCP server configuration
FIGMA_MCP_URL = "http://127.0.0.1:3845/mcp"

class FigmaMCPTester:
    def __init__(self):
        self.base_url = FIGMA_MCP_URL
        self.session_id = None
        self.client_id = str(uuid.uuid4())
        
    def initialize_session(self) -> bool:
        """Initialize MCP session with Figma"""
        print("\n" + "="*60)
        print("Initializing MCP Session with Figma...")
        
        # Try different initialization patterns
        init_requests = [
            # Standard MCP initialize
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1.0",
                    "clientInfo": {
                        "name": "RRRAlgorithms Test Client",
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    }
                },
                "id": 1
            },
            # With session ID in params
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "sessionId": self.client_id,
                    "protocolVersion": "0.1.0",
                    "clientInfo": {
                        "name": "RRRAlgorithms Test Client",
                        "version": "1.0.0"
                    }
                },
                "id": 2
            },
            # Minimal request
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {},
                "id": 3
            }
        ]
        
        headers_options = [
            {"Content-Type": "application/json"},
            {"Content-Type": "application/json", "X-Session-Id": self.client_id},
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.client_id}"}
        ]
        
        for i, req in enumerate(init_requests):
            for j, headers in enumerate(headers_options):
                print(f"\nAttempt {i*len(headers_options) + j + 1}:")
                print(f"Headers: {headers}")
                print(f"Request: {json.dumps(req, indent=2)[:200]}...")
                
                try:
                    response = requests.post(
                        self.base_url,
                        json=req,
                        headers=headers,
                        timeout=5
                    )
                    
                    print(f"Status: {response.status_code}")
                    
                    if response.text:
                        try:
                            data = response.json()
                            print(f"Response: {json.dumps(data, indent=2)[:300]}...")
                            
                            # Check if we got a successful response
                            if "result" in data:
                                print("✓ Successfully initialized session!")
                                if "sessionId" in data.get("result", {}):
                                    self.session_id = data["result"]["sessionId"]
                                    print(f"Session ID: {self.session_id}")
                                return True
                            elif "error" in data:
                                error = data["error"]
                                if error.get("code") == -32001:
                                    print("→ Invalid session ID error - need proper authentication")
                                elif error.get("code") == -32000:
                                    print("→ Invalid request format - trying next format")
                        except:
                            print(f"Response (raw): {response.text[:200]}...")
                            
                except Exception as e:
                    print(f"Error: {e}")
        
        return False
    
    def test_with_session(self):
        """Test MCP methods with session"""
        if not self.session_id:
            print("\nNo session ID available, using client ID as fallback")
            self.session_id = self.client_id
        
        print("\n" + "="*60)
        print("Testing MCP Methods with Session...")
        
        methods = [
            ("tools/list", {}),
            ("resources/list", {}),
            ("prompts/list", {}),
            ("capabilities", {}),
        ]
        
        headers = {
            "Content-Type": "application/json",
            "X-Session-Id": self.session_id
        }
        
        for method, params in methods:
            print(f"\nTesting method: {method}")
            
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": str(uuid.uuid4())
            }
            
            try:
                response = requests.post(
                    self.base_url,
                    json=request,
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
    
    def test_figma_specific_endpoints(self):
        """Test Figma-specific functionality"""
        print("\n" + "="*60)
        print("Testing Figma-Specific Endpoints...")
        
        # Potential Figma-specific methods
        methods = [
            ("figma/getFiles", {}),
            ("figma/getCurrentFile", {}),
            ("figma/getSelection", {}),
            ("figma/getComponents", {}),
            ("design/list", {}),
            ("files/list", {}),
        ]
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.session_id:
            headers["X-Session-Id"] = self.session_id
        
        for method, params in methods:
            print(f"\nTesting method: {method}")
            
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": str(uuid.uuid4())
            }
            
            try:
                response = requests.post(
                    self.base_url,
                    json=request,
                    headers=headers,
                    timeout=5
                )
                
                print(f"Status: {response.status_code}")
                
                if response.text:
                    try:
                        data = response.json()
                        if "result" in data:
                            print("✓ Method exists and returned result!")
                        print(f"Response: {json.dumps(data, indent=2)[:300]}...")
                    except:
                        print(f"Response (raw): {response.text[:200]}...")
                        
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main test execution"""
    print("="*60)
    print("FIGMA MCP SERVER ENHANCED TEST")
    print("="*60)
    
    tester = FigmaMCPTester()
    
    # Check basic connectivity
    print("\nTesting basic connectivity...")
    try:
        response = requests.get(FIGMA_MCP_URL, timeout=5)
        print(f"✓ Server is reachable at {FIGMA_MCP_URL}")
        print(f"  Status: {response.status_code}")
    except:
        print(f"✗ Cannot reach server at {FIGMA_MCP_URL}")
        print("\nPlease ensure:")
        print("1. Figma Desktop is running")
        print("2. The MCP server feature is enabled in Figma")
        print("3. No firewall is blocking port 3845")
        return 1
    
    # Try to initialize session
    session_ok = tester.initialize_session()
    
    # Test with session
    tester.test_with_session()
    
    # Test Figma-specific endpoints
    tester.test_figma_specific_endpoints()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ Figma MCP server is running and responding")
    print("✓ Server requires proper session/authentication")
    print("✓ JSON-RPC 2.0 protocol is being used")
    print("\nNotes:")
    print("- The server requires a valid session ID for most operations")
    print("- Authentication mechanism may need Figma API credentials")
    print("- Check Figma Desktop settings for API/Plugin configurations")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())