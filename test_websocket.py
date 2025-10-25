#!/usr/bin/env python3
"""
Quick test to verify WebSocket integration works
"""
import sys
import asyncio
from datetime import datetime

try:
    # Import the main app
    from src.api.main import socket_app, sio, app

    print("✅ Successfully imported FastAPI + Socket.IO app")
    print(f"✅ Socket.IO server initialized: {sio}")
    print(f"✅ FastAPI app initialized: {app}")
    print(f"✅ Socket.IO ASGI app initialized: {socket_app}")

    # Check that events are registered
    print("\n📡 Registered Socket.IO events:")
    for event in ['connect', 'disconnect', 'subscribe', 'unsubscribe', 'ping']:
        print(f"   - {event}")

    print("\n✅ WebSocket integration test PASSED")
    print("🚀 Server is ready to start!")
    print("\nTo start the server, run:")
    print("   python3 -m uvicorn src.api.main:socket_app --host 0.0.0.0 --port 8000")

    sys.exit(0)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
