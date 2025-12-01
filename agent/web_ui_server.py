import os
import sys
import uuid
import json
import asyncio
import time
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import config
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from customer_auth import CustomerAuthenticator

# Import performance tracking
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from performance_metrics import PerformanceMetrics, performance_tracker
    PERFORMANCE_TRACKING_ENABLED = True
    print("✅ Performance tracking enabled")
except ImportError:
    PERFORMANCE_TRACKING_ENABLED = False
    print("⚠️  Performance tracking disabled (performance_metrics.py not found)")

# Initialize FastAPI app
app = FastAPI(title="Secure Banking Assistant")

# Store active sessions
active_sessions = {}

# Global session service shared by all ChatSession instances
global_session_service = InMemorySessionService()

class ChatSession:
    def __init__(self, user_data, user_type, session_token):
        self.user_data = user_data
        self.user_type = user_type
        self.session_token = session_token
        self.conversation_history = []
        
        if user_type == 'customer':
            self.current_user = user_data['name']
            self.current_user_vpa = user_data['primary_vpa']
            self.customer_id = user_data.get('customer_id')
        elif user_type == 'merchant':
            self.current_user = user_data['merchant_vpa']
            self.current_user_vpa = user_data['merchant_vpa']
            self.merchant_id = user_data.get('merchant_id')
            self.merchant_name = user_data.get('merchant_name')
        
        # Use a deterministic session ID based on username
        # This ensures the same user always gets the same ADK session
        self.adk_session_id = f"session_{self.current_user.replace('@', '_').replace('.', '_')}"
        
        print(f"   ADK Session ID: {self.adk_session_id}")
        
        self.mcp_tools = MCPToolset(
            connection_params=SseServerParams(
                url="http://localhost:8001/sse"
            )
        )
        
        self.agent_instruction = self._build_instruction()
        
        self.agent = Agent(
            name="secure_banking_agent",
            model=Gemini(
                model_name="gemini-2.5-flash",
                project=config.GCP_PROJECT_ID,
                location=config.GCP_LOCATION,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "text/plain"
                },
                # Enable streaming at model level
                stream=True
            ),
            instruction=self.agent_instruction,
            tools=[self.mcp_tools]
        )
        
        # Don't create runner here - create it per message to avoid session issues
        print(f"   ✅ Agent initialized")
    
    def _build_instruction(self):
        if self.user_type == 'customer':
            user_intro = f"AUTHENTICATED CUSTOMER: {self.current_user}\nVPA: {self.current_user_vpa}"
        else:
            user_intro = f"AUTHENTICATED MERCHANT: {self.merchant_name}\nVPA: {self.current_user_vpa}"
        
        return (
            f"You are a banking assistant.\n{user_intro}\n"
            f"ALWAYS use current_user='{self.current_user}' and user_type='{self.user_type}' when calling tools."
        )
    
    async def send_message(self, message: str):
        """Send a message and get streaming response with performance tracking"""
        responses = []
        
        # Performance tracking
        query_start_time = time.time()
        agent_start_time = None
        first_chunk_time = None
        
        # Create runner with its own session service
        if not hasattr(self, 'runner'):
            self.session_service = InMemorySessionService()
            self.runner = Runner(
                app_name="agent",
                agent=self.agent,
                session_service=self.session_service
            )
            
            # Create the session SYNCHRONOUSLY before first use
            print(f"   Creating initial session synchronously...")
            self.session_service.create_session_sync(
                app_name="agent",
                user_id=self.current_user,
                session_id=self.adk_session_id
            )
            print(f"   ✓ Initial session created")
        
        print(f"📤 Sending to ADK - User: {self.current_user}, Session: {self.adk_session_id}")
        
        try:
            chunk_count = 0
            agent_start_time = time.time()
            
            async for event in self.runner.run_async(
                user_id=self.current_user,
                session_id=self.adk_session_id,
                new_message=Content(parts=[Part(text=message)])
            ):
                chunk_count += 1
                
                if hasattr(event, 'content') and event.content:
                    # Check if content has parts
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            # Handle text parts (agent's natural language response)
                            if hasattr(part, 'text') and part.text:
                                chunk = part.text
                                
                                # Track time to first chunk
                                if first_chunk_time is None:
                                    first_chunk_time = time.time()
                                    time_to_first_chunk = first_chunk_time - query_start_time
                                    print(f"⚡ Time to first chunk: {time_to_first_chunk:.3f}s")
                                
                                # CHARACTER-BY-CHARACTER STREAMING for maximum speed
                                # Stream every 3-5 characters for smooth typing effect
                                chunk_size = 3  # Adjust this: 1=slowest/smoothest, 5=faster, 10=very fast
                                
                                for i in range(0, len(chunk), chunk_size):
                                    mini_chunk = chunk[i:i+chunk_size]
                                    responses.append(mini_chunk)
                                    yield mini_chunk
                                    # Optional: tiny delay for ultra-smooth effect
                                    # await asyncio.sleep(0.005)  # 5ms delay
                            
                            # Handle function responses (tool results)
                            elif hasattr(part, 'function_response'):
                                # Tool result - stream character by character
                                func_response = part.function_response
                                if hasattr(func_response, 'response'):
                                    response_data = func_response.response
                                    
                                    if isinstance(response_data, dict):
                                        if 'content' in response_data:
                                            content = response_data['content']
                                            if isinstance(content, list):
                                                for item in content:
                                                    if isinstance(item, dict) and 'text' in item:
                                                        result_text = item['text']
                                                        # Stream in small character chunks
                                                        chunk_size = 3
                                                        for i in range(0, len(result_text), chunk_size):
                                                            mini_chunk = result_text[i:i+chunk_size]
                                                            responses.append(mini_chunk)
                                                            yield mini_chunk
                    
                    # Fallback: direct text
                    elif hasattr(event.content, 'text') and event.content.text:
                        chunk = event.content.text
                        
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            time_to_first_chunk = first_chunk_time - query_start_time
                            print(f"⚡ Time to first chunk: {time_to_first_chunk:.3f}s")
                        
                        # Stream in small character chunks
                        chunk_size = 3
                        for i in range(0, len(chunk), chunk_size):
                            mini_chunk = chunk[i:i+chunk_size]
                            responses.append(mini_chunk)
                            yield mini_chunk
            
            # Calculate timing metrics
            total_time = time.time() - query_start_time
            agent_processing_time = time.time() - agent_start_time if agent_start_time else 0
            
            print(f"✅ Streaming complete. Total chunks: {len(responses)}")
            print(f"⏱️  Performance Summary:")
            print(f"   • Total Time: {total_time:.3f}s")
            print(f"   • Agent Processing: {agent_processing_time:.3f}s")
            if first_chunk_time:
                print(f"   • Time to First Chunk: {(first_chunk_time - query_start_time):.3f}s")
            
            # Log performance metric
            if PERFORMANCE_TRACKING_ENABLED:
                metric = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    user=self.current_user,
                    user_type=self.user_type,
                    query=message[:200],
                    total_time=total_time,
                    agent_response_time=agent_processing_time,
                    tools_used=['send_message'],
                    tool_count=1,
                    status='SUCCESS'
                )
                performance_tracker.log_metric(metric)
            
        except Exception as e:
            total_time = time.time() - query_start_time
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Log error metric
            if PERFORMANCE_TRACKING_ENABLED:
                metric = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    user=self.current_user,
                    user_type=self.user_type,
                    query=message[:200],
                    total_time=total_time,
                    status='ERROR',
                    error_message=str(e)
                )
                performance_tracker.log_metric(metric)
            
            yield f"Error: {str(e)}"
        
        full_response = "".join(responses)
        if full_response:
            self.conversation_history.append({
                "user": message,
                "assistant": full_response,
                "timestamp": datetime.now().isoformat()
            })

@app.post("/api/authenticate")
async def authenticate(
    vpa: str = Form(...),
    pin_or_password: str = Form(...),
    user_type: str = Form(...),
    auth_method: str = Form(default="vpa")
):
    try:
        print(f"\n{'='*60}")
        print(f"🔐 Authentication attempt")
        print(f"   Type: {user_type}")
        print(f"   Identifier: {vpa}")
        print(f"{'='*60}")
        
        authenticator = CustomerAuthenticator()
        user_data = None
        
        if user_type == 'customer':
            if auth_method == 'vpa':
                user_data = authenticator.authenticate_by_vpa_pin(vpa, pin_or_password)
            else:
                user_data = authenticator.authenticate_by_mobile_pin(vpa, pin_or_password)
            
            if not user_data:
                print(f"❌ Customer authentication failed")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid credentials"}
                )
            print(f"✅ Customer authenticated: {user_data['name']}")
        
        elif user_type == 'merchant':
            user_data = authenticator.authenticate_merchant_by_vpa_password(vpa, pin_or_password)
            if not user_data:
                print(f"❌ Merchant authentication failed")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid credentials"}
                )
            print(f"✅ Merchant authenticated: {user_data['merchant_name']}")
        
        # Generate session token BEFORE creating ChatSession
        session_token = str(uuid.uuid4())
        print(f"📝 Creating session with token: {session_token}")
        
        session = ChatSession(user_data, user_type, session_token)
        active_sessions[session_token] = session
        
        print(f"✅ Session created and stored")
        print(f"   Total active sessions: {len(active_sessions)}")
        print(f"   Session keys: {list(active_sessions.keys())}")
        print(f"{'='*60}\n")
        
        return {
            "session_token": session_token,
            "user_type": user_type,
            "user_name": session.current_user,
            "user_vpa": session.current_user_vpa
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/sample-merchants")
async def get_sample_merchants():
    try:
        authenticator = CustomerAuthenticator()
        merchants = authenticator.get_sample_merchants(limit=5)
        
        merchant_list = []
        for m in merchants:
            merchant_list.append({
                "vpa": m['merchant_vpa'],
                "name": m['merchant_name'],
                "category": m['category'] if m['category'] else 'Other',
                "password": m['password']
            })
        
        return {"merchants": merchant_list}
    except Exception as e:
        return {"merchants": [], "error": str(e)}

@app.websocket("/ws/chat/{session_token}")
async def chat_websocket(websocket: WebSocket, session_token: str):
    await websocket.accept()
    
    print(f"\n🔌 WebSocket connection attempt")
    print(f"   Session token: {session_token}")
    print(f"   Active sessions: {list(active_sessions.keys())}")
    
    session = active_sessions.get(session_token)
    if not session:
        error_msg = f"Session not found: {session_token}"
        print(f"❌ {error_msg}")
        print(f"   Available sessions: {len(active_sessions)}")
        await websocket.send_json({"type": "error", "message": error_msg})
        await websocket.close()
        return
    
    print(f"✅ Session found: {session.current_user}")
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                continue
            
            print(f"📨 Message from {session.current_user}: {message[:50]}...")
            
            await websocket.send_json({"type": "status"})
            
            async for chunk in session.send_message(message):
                await websocket.send_json({"type": "response", "chunk": chunk})
            
            await websocket.send_json({"type": "complete"})
            
    except WebSocketDisconnect:
        print(f"🔌 Client disconnected: {session_token}")
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()

@app.get("/", response_class=HTMLResponse)
async def get_chat_ui():
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Banking Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            width: 90%;
            max-width: 900px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
        }
        .auth-container { padding: 40px; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: bold; }
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        .auth-button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        .error-message {
            color: red;
            margin-top: 10px;
            display: none;
        }
        .error-message.show { display: block; }
        .sample-merchants {
            margin-top: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            display: none;
        }
        .sample-merchants.show { display: block; }
        .merchant-item {
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            cursor: pointer;
        }
        .merchant-item:hover { background: #e0e0ff; }
        .chat-container { flex: 1; display: none; flex-direction: column; }
        .chat-container.show { display: flex; }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 70%;
            padding: 12px;
            border-radius: 15px;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.assistant .message-content {
            background: white;
        }
        .input-container {
            padding: 20px;
            background: white;
            display: flex;
            gap: 10px;
        }
        .input-container input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 20px;
            font-size: 16px;
        }
        .input-container button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 20px;
            cursor: pointer;
        }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Secure Banking Assistant</h1>
            <div id="userInfo"></div>
        </div>
        
        <div class="auth-container" id="authContainer">
            <div class="form-group">
                <label>User Type</label>
                <select id="userType">
                    <option value="customer">Customer</option>
                    <option value="merchant">Merchant</option>
                </select>
            </div>
            
            <div class="form-group">
                <label id="identifierLabel">VPA</label>
                <input type="text" id="identifier" placeholder="example@okicici" />
            </div>
            
            <div class="form-group">
                <label id="pinLabel">PIN</label>
                <input type="password" id="pinOrPassword" placeholder="Enter PIN" />
            </div>
            
            <button class="auth-button" id="authButton">🔑 Authenticate</button>
            
            <div class="error-message" id="errorMessage"></div>
            
            <div class="sample-merchants" id="sampleMerchants">
                <h4>Sample Merchants:</h4>
                <div id="merchantsList"></div>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="messages" id="messages"></div>
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Ask me anything..." />
                <button id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        var ws = null;
        var sessionToken = null;
        var currentMessage = '';
        
        document.getElementById('userType').onchange = function() {
            var isCustomer = this.value === 'customer';
            document.getElementById('identifierLabel').textContent = isCustomer ? 'VPA' : 'Merchant VPA';
            document.getElementById('pinLabel').textContent = isCustomer ? 'PIN' : 'Password';
            document.getElementById('identifier').placeholder = isCustomer ? 'example@okicici' : 'merchant@okaxis';
            
            if (!isCustomer) {
                fetch('/api/sample-merchants')
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        var list = document.getElementById('merchantsList');
                        list.innerHTML = '';
                        if (data.merchants) {
                            data.merchants.forEach(function(m) {
                                var div = document.createElement('div');
                                div.className = 'merchant-item';
                                div.textContent = m.vpa + ' - ' + m.name + ' (Password: ' + m.password + ')';
                                div.onclick = function() {
                                    document.getElementById('identifier').value = m.vpa;
                                    document.getElementById('pinOrPassword').value = m.password;
                                };
                                list.appendChild(div);
                            });
                            document.getElementById('sampleMerchants').classList.add('show');
                        }
                    });
            } else {
                document.getElementById('sampleMerchants').classList.remove('show');
            }
        };
        
        document.getElementById('authButton').onclick = function() {
            var vpa = document.getElementById('identifier').value;
            var pass = document.getElementById('pinOrPassword').value;
            var type = document.getElementById('userType').value;
            var btn = this;
            var errDiv = document.getElementById('errorMessage');
            
            errDiv.classList.remove('show');
            btn.disabled = true;
            btn.textContent = 'Authenticating...';
            
            var fd = new FormData();
            fd.append('vpa', vpa);
            fd.append('pin_or_password', pass);
            fd.append('user_type', type);
            fd.append('auth_method', 'vpa');
            
            fetch('/api/authenticate', { method: 'POST', body: fd })
                .then(function(r) { return r.json().then(function(d) { return {status: r.status, data: d}; }); })
                .then(function(result) {
                    if (result.status === 200) {
                        sessionToken = result.data.session_token;
                        document.getElementById('userInfo').textContent = result.data.user_name;
                        document.getElementById('authContainer').classList.add('hidden');
                        document.getElementById('chatContainer').classList.add('show');
                        connectWS();
                        addMsg('assistant', 'Welcome! How can I help you?');
                    } else {
                        errDiv.textContent = result.data.error;
                        errDiv.classList.add('show');
                        btn.disabled = false;
                        btn.textContent = '🔑 Authenticate';
                    }
                });
        };
        
        function connectWS() {
            var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws/chat/' + sessionToken);
            ws.onmessage = function(e) {
                var d = JSON.parse(e.data);
                if (d.type === 'response') {
                    currentMessage += d.chunk;
                    updateLast(currentMessage);
                } else if (d.type === 'complete') {
                    currentMessage = '';
                    document.getElementById('sendButton').disabled = false;
                }
            };
        }
        
        document.getElementById('sendButton').onclick = function() {
            var inp = document.getElementById('messageInput');
            var msg = inp.value.trim();
            if (!msg || !ws) return;
            addMsg('user', msg);
            inp.value = '';
            this.disabled = true;
            currentMessage = '';
            addMsg('assistant', '');
            ws.send(JSON.stringify({message: msg}));
        };
        
        document.getElementById('messageInput').onkeypress = function(e) {
            if (e.key === 'Enter') document.getElementById('sendButton').click();
        };
        
        function addMsg(type, content) {
            var div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerHTML = '<div class="message-content">' + content.replace(/\\n/g, '<br>') + '</div>';
            document.getElementById('messages').appendChild(div);
            document.getElementById('messages').scrollTop = 999999;
        }
        
        function updateLast(content) {
            var msgs = document.getElementById('messages');
            var last = msgs.lastElementChild;
            if (last) {
                last.querySelector('.message-content').innerHTML = content.replace(/\\n/g, '<br>');
                msgs.scrollTop = 999999;
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Web UI Server")
    print("=" * 60)
    print(f"🌍 URL: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)