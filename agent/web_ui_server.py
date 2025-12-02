import os
import sys
import uuid
import json
import asyncio
import time
import re
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

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

# --- Performance Metrics Setup ---
@dataclass
class PerformanceMetrics:
    timestamp: str
    user: str
    user_type: str
    query: str
    total_time: float
    agent_response_time: float
    time_to_sql_display: Optional[float] = None  # Time until SQL shown
    time_to_first_chunk: Optional[float] = None  # Time until results start streaming
    streaming_duration: Optional[float] = None   # How long streaming took
    status: str = "SUCCESS"
    error_message: str = None

perf_logger = logging.getLogger('web_ui_performance')
perf_logger.setLevel(logging.INFO)
log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
perf_log_path = os.path.join(log_dir, 'performance_metrics.log')
perf_handler = logging.FileHandler(perf_log_path)
perf_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
perf_logger.addHandler(perf_handler)

app = FastAPI(title="Secure Banking Assistant")
active_sessions = {}


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
        
        self.adk_session_id = f"session_{self.current_user.replace('@', '_').replace('.', '_')}"
        
        self.mcp_tools = MCPToolset(
            connection_params=SseServerParams(url="http://localhost:8001/sse")
        )
        
        self.agent = Agent(
            name="secure_banking_agent",
            model=Gemini(
                model_name="gemini-2.5-flash",
                project=config.GCP_PROJECT_ID,
                location=config.GCP_LOCATION,
                thinking_budget=0,
                generation_config={"temperature": 0, "response_mime_type": "text/plain"},
                stream=True
            ),
            instruction=self._build_instruction(),
            tools=[self.mcp_tools]
        )
        print(f"   ✅ Agent initialized for {self.current_user}")
    
    def _build_instruction(self):
        if self.user_type == 'customer':
            user_intro = f"AUTHENTICATED CUSTOMER: {self.current_user}\nVPA: {self.current_user_vpa}"
        else:
            user_intro = f"AUTHENTICATED MERCHANT: {self.merchant_name}\nVPA: {self.current_user_vpa}"

        return (
            f"You are a banking assistant.\n{user_intro}\n"
            f"ALWAYS use current_user='{self.current_user}' and user_type='{self.user_type}' when calling tools.\n\n"
            f"WORKFLOW:\n"
            f"1. Call generate_sql_for_query → returns raw SQL\n"
            f"2. Wrap SQL in ```sql ... ``` code block\n"
            f"3. Call execute_sql_query → returns formatted transactions\n"
            f"4. Display transactions as received\n\n"
            f"RULES: Display SQL ONCE in code block. Display results ONCE. No extra text."
        )
    
    async def send_message(self, message: str):
        """Stream response in real-time with granular performance tracking"""
        
        # Start performance tracking
        start_time = time.time()
        agent_start_time = None
        sql_display_time = None
        first_chunk_time = None
        streaming_start_time = None
        
        if not hasattr(self, 'runner'):
            self.session_service = InMemorySessionService()
            self.runner = Runner(
                app_name="agent",
                agent=self.agent,
                session_service=self.session_service
            )
            self.session_service.create_session_sync(
                app_name="agent",
                user_id=self.current_user,
                session_id=self.adk_session_id
            )
        
        print(f"\n{'='*60}")
        print(f"📤 Query: {message[:50]}...")
        print(f"{'='*60}")
        
        full_response = []
        event_count = 0
        
        try:
            agent_start_time = time.time()
            async for event in self.runner.run_async(
                user_id=self.current_user,
                session_id=self.adk_session_id,
                new_message=Content(parts=[Part(text=message)])
            ):
                event_count += 1
                
                # Debug: Log event type and structure
                event_type = type(event).__name__
                print(f"\n🔍 Event #{event_count}: {event_type}")
                
                # Check for different event attributes
                if hasattr(event, 'is_final_response'):
                    print(f"   is_final_response: {event.is_final_response()}")
                
                if hasattr(event, 'author'):
                    print(f"   author: {event.author}")
                
                # Method 1: Check for content attribute
                if hasattr(event, 'content') and event.content:
                    print(f"   📦 Has content attribute")
                    
                    # Handle Content object with parts
                    if hasattr(event.content, 'parts') and event.content.parts:
                        print(f"   📋 Content has {len(event.content.parts)} parts")
                        
                        for i, part in enumerate(event.content.parts):
                            part_type = type(part).__name__
                            print(f"      Part {i}: {part_type}")
                            
                            # Skip function calls and responses
                            if hasattr(part, 'function_call') and part.function_call:
                                fn_name = part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'
                                print(f"         ⏭️ Skipping function_call: {fn_name}")
                                continue
                            
                            if hasattr(part, 'function_response') and part.function_response:
                                print(f"         ⏭️ Skipping function_response")
                                continue
                            
                            # Extract text from part
                            if hasattr(part, 'text') and part.text:
                                chunk = part.text
                                print(f"         ✅ Text ({len(chunk)} chars): {chunk[:100]}...")
                                
                                # Detect SQL display
                                if sql_display_time is None and '```sql' in chunk.lower():
                                    sql_display_time = time.time()
                                    print(f"         🔍 SQL DISPLAYED at {sql_display_time - start_time:.3f}s")
                                
                                # Detect first data chunk (not SQL)
                                if first_chunk_time is None and '```sql' not in chunk.lower() and len(chunk.strip()) > 0:
                                    first_chunk_time = time.time()
                                    streaming_start_time = first_chunk_time
                                    print(f"         📊 FIRST CHUNK at {first_chunk_time - start_time:.3f}s")
                                
                                # Stream in small pieces
                                for j in range(0, len(chunk), 10):
                                    mini = chunk[j:j+10]
                                    full_response.append(mini)
                                    yield mini
                                    await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
                # Method 2: Check for text attribute directly on event
                elif hasattr(event, 'text') and event.text:
                    chunk = event.text
                    print(f"   📝 Direct text on event ({len(chunk)} chars): {chunk[:100]}...")
                    
                    # Detect SQL display
                    if sql_display_time is None and '```sql' in chunk.lower():
                        sql_display_time = time.time()
                        print(f"   🔍 SQL DISPLAYED at {sql_display_time - start_time:.3f}s")
                    
                    # Detect first data chunk
                    if first_chunk_time is None and '```sql' not in chunk.lower() and len(chunk.strip()) > 0:
                        first_chunk_time = time.time()
                        streaming_start_time = first_chunk_time
                        print(f"   📊 FIRST CHUNK at {first_chunk_time - start_time:.3f}s")
                    
                    for j in range(0, len(chunk), 10):
                        mini = chunk[j:j+10]
                        full_response.append(mini)
                        yield mini
                        await asyncio.sleep(0.01)
                
                # Method 3: Check for response attribute (some ADK versions)
                elif hasattr(event, 'response') and event.response:
                    print(f"   📝 Has response attribute")
                    response_obj = event.response
                    
                    if hasattr(response_obj, 'text') and response_obj.text:
                        chunk = response_obj.text
                        print(f"      Text from response ({len(chunk)} chars): {chunk[:100]}...")
                        
                        # Detect SQL display
                        if sql_display_time is None and '```sql' in chunk.lower():
                            sql_display_time = time.time()
                            print(f"      🔍 SQL DISPLAYED at {sql_display_time - start_time:.3f}s")
                        
                        # Detect first data chunk
                        if first_chunk_time is None and '```sql' not in chunk.lower() and len(chunk.strip()) > 0:
                            first_chunk_time = time.time()
                            streaming_start_time = first_chunk_time
                            print(f"      📊 FIRST CHUNK at {first_chunk_time - start_time:.3f}s")
                        
                        for j in range(0, len(chunk), 10):
                            mini = chunk[j:j+10]
                            full_response.append(mini)
                            yield mini
                            await asyncio.sleep(0.01)
                    
                    elif hasattr(response_obj, 'candidates') and response_obj.candidates:
                        print(f"      Has {len(response_obj.candidates)} candidates")
                        for candidate in response_obj.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            chunk = part.text
                                            print(f"      Candidate text ({len(chunk)} chars): {chunk[:100]}...")
                                            
                                            # Detect SQL display
                                            if sql_display_time is None and '```sql' in chunk.lower():
                                                sql_display_time = time.time()
                                                print(f"      🔍 SQL DISPLAYED at {sql_display_time - start_time:.3f}s")
                                            
                                            # Detect first data chunk
                                            if first_chunk_time is None and '```sql' not in chunk.lower() and len(chunk.strip()) > 0:
                                                first_chunk_time = time.time()
                                                streaming_start_time = first_chunk_time
                                                print(f"      📊 FIRST CHUNK at {first_chunk_time - start_time:.3f}s")
                                            
                                            for j in range(0, len(chunk), 10):
                                                mini = chunk[j:j+10]
                                                full_response.append(mini)
                                                yield mini
                                                await asyncio.sleep(0.01)
                
                # Method 4: Try to access as dict
                elif hasattr(event, '__dict__'):
                    event_dict = event.__dict__
                    print(f"   📝 Event dict keys: {list(event_dict.keys())}")
                    
                    # Look for any text-like attributes
                    for key in ['text', 'message', 'output', 'result']:
                        if key in event_dict and event_dict[key]:
                            val = event_dict[key]
                            if isinstance(val, str):
                                print(f"      Found {key}: {val[:100]}...")
                                
                                # Detect SQL display
                                if sql_display_time is None and '```sql' in val.lower():
                                    sql_display_time = time.time()
                                    print(f"      🔍 SQL DISPLAYED at {sql_display_time - start_time:.3f}s")
                                
                                # Detect first data chunk
                                if first_chunk_time is None and '```sql' not in val.lower() and len(val.strip()) > 0:
                                    first_chunk_time = time.time()
                                    streaming_start_time = first_chunk_time
                                    print(f"      📊 FIRST CHUNK at {first_chunk_time - start_time:.3f}s")
                                
                                for j in range(0, len(val), 10):
                                    mini = val[j:j+10]
                                    full_response.append(mini)
                                    yield mini
                                    await asyncio.sleep(0.01)
                
                else:
                    print(f"   ⚠️ Unknown event structure")
                    # Try string conversion as last resort
                    try:
                        event_str = str(event)
                        if len(event_str) > 0 and event_str != str(type(event)):
                            print(f"   String repr: {event_str[:200]}...")
                    except:
                        pass
            
            # Final summary
            final_response = "".join(full_response)
            total_time = time.time() - start_time
            agent_time = time.time() - agent_start_time if agent_start_time else total_time
            
            # Calculate streaming duration
            streaming_duration = None
            if streaming_start_time:
                streaming_duration = time.time() - streaming_start_time
            
            print(f"\n{'='*60}")
            print(f"✅ Streaming complete")
            print(f"   Total events: {event_count}")
            print(f"   Response length: {len(final_response)} chars")
            print(f"   Total time: {total_time:.3f}s")
            if sql_display_time:
                print(f"   Time to SQL display: {sql_display_time - start_time:.3f}s")
            if first_chunk_time:
                print(f"   Time to first chunk: {first_chunk_time - start_time:.3f}s")
            if streaming_duration:
                print(f"   Streaming duration: {streaming_duration:.3f}s")
            print(f"{'='*60}\n")
            
            # Log performance metrics with granular timestamps
            metric = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                user=self.current_user,
                user_type=self.user_type,
                query=message[:200],
                total_time=total_time,
                agent_response_time=agent_time,
                time_to_sql_display=sql_display_time - start_time if sql_display_time else None,
                time_to_first_chunk=first_chunk_time - start_time if first_chunk_time else None,
                streaming_duration=streaming_duration,
                status="SUCCESS"
            )
            perf_logger.info(json.dumps(asdict(metric)))
            
            if final_response:
                self.conversation_history.append({
                    "user": message,
                    "assistant": final_response,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print("⚠️ Warning: Empty response after processing all events!")
                # Yield a fallback message
                fallback = "I processed your request but couldn't generate a response. Please try again."
                yield fallback
            
        except Exception as e:
            total_time = time.time() - start_time
            
            # Log error metric
            metric = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                user=self.current_user,
                user_type=self.user_type,
                query=message[:200],
                total_time=total_time,
                agent_response_time=0,
                time_to_sql_display=sql_display_time - start_time if sql_display_time else None,
                time_to_first_chunk=first_chunk_time - start_time if first_chunk_time else None,
                streaming_duration=None,
                status="ERROR",
                error_message=str(e)[:200]
            )
            perf_logger.info(json.dumps(asdict(metric)))
            
            print(f"❌ Error in send_message: {e}")
            import traceback
            traceback.print_exc()
            yield f"Error: {str(e)}"


@app.post("/api/authenticate")
async def authenticate(
    vpa: str = Form(...),
    pin_or_password: str = Form(...),
    user_type: str = Form(...),
    auth_method: str = Form(default="vpa")
):
    try:
        authenticator = CustomerAuthenticator()
        user_data = None
        
        if user_type == 'customer':
            user_data = authenticator.authenticate_by_vpa_pin(vpa, pin_or_password) if auth_method == 'vpa' else authenticator.authenticate_by_mobile_pin(vpa, pin_or_password)
            if not user_data:
                return JSONResponse(status_code=401, content={"error": "Invalid credentials"})
        elif user_type == 'merchant':
            user_data = authenticator.authenticate_merchant_by_vpa_password(vpa, pin_or_password)
            if not user_data:
                return JSONResponse(status_code=401, content={"error": "Invalid credentials"})
        
        session_token = str(uuid.uuid4())
        session = ChatSession(user_data, user_type, session_token)
        active_sessions[session_token] = session
        
        return {
            "session_token": session_token,
            "user_type": user_type,
            "user_name": session.current_user,
            "user_vpa": session.current_user_vpa
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/sample-merchants")
async def get_sample_merchants():
    try:
        authenticator = CustomerAuthenticator()
        merchants = authenticator.get_sample_merchants(limit=5)
        return {"merchants": [
            {"vpa": m['merchant_vpa'], "name": m['merchant_name'], 
             "category": m['category'] or 'Other', "password": m['password']}
            for m in merchants
        ]}
    except:
        return {"merchants": []}


@app.websocket("/ws/chat/{session_token}")
async def chat_websocket(websocket: WebSocket, session_token: str):
    await websocket.accept()
    print(f"🔌 WebSocket connected: {session_token}")
    
    session = active_sessions.get(session_token)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            if not message:
                continue
            
            print(f"📨 Received message: {message[:50]}...")
            await websocket.send_json({"type": "status"})
            
            chunk_count = 0
            async for chunk in session.send_message(message):
                chunk_count += 1
                await websocket.send_json({"type": "response", "chunk": chunk})
                # Don't add extra delay here - the send_message already has delays
            
            print(f"📤 Sent {chunk_count} chunks to client")
            await websocket.send_json({"type": "complete"})
            
    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_token}")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        import traceback
        traceback.print_exc()


@app.get("/", response_class=HTMLResponse)
async def get_chat_ui():
    return HTMLResponse(content='''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Banking Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { width: 90%; max-width: 1000px; height: 90vh; background: white; border-radius: 20px; display: flex; flex-direction: column; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 20px 20px 0 0; }
        .auth-container { padding: 40px; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: bold; }
        .form-group input, .form-group select { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
        .auth-button { width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px; border-radius: 8px; font-size: 16px; cursor: pointer; }
        .error-message { color: red; margin-top: 10px; display: none; }
        .error-message.show { display: block; }
        .sample-merchants { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px; display: none; }
        .sample-merchants.show { display: block; }
        .merchant-item { padding: 10px; margin: 5px 0; background: white; border-radius: 5px; cursor: pointer; }
        .merchant-item:hover { background: #e0e0ff; }
        .chat-container { flex: 1; display: none; flex-direction: column; }
        .chat-container.show { display: flex; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; background: #f5f7fa; }
        .message { margin-bottom: 15px; display: flex; }
        .message.user { justify-content: flex-end; }
        .message-content { max-width: 85%; padding: 15px; border-radius: 15px; white-space: pre-wrap; font-family: monospace; font-size: 13px; line-height: 1.6; }
        .message.user .message-content { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .message.assistant .message-content { background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .input-container { padding: 20px; background: white; display: flex; gap: 10px; }
        .input-container input { flex: 1; padding: 15px; border: 2px solid #eee; border-radius: 25px; font-size: 16px; }
        .input-container button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 30px; border-radius: 25px; cursor: pointer; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>🔐 Secure Banking Assistant</h1><div id="userInfo"></div></div>
        <div class="auth-container" id="authContainer">
            <div class="form-group"><label>User Type</label><select id="userType"><option value="customer">Customer</option><option value="merchant">Merchant</option></select></div>
            <div class="form-group"><label id="identifierLabel">VPA</label><input type="text" id="identifier" placeholder="example@okicici" /></div>
            <div class="form-group"><label id="pinLabel">PIN</label><input type="password" id="pinOrPassword" /></div>
            <button class="auth-button" id="authButton">🔑 Authenticate</button>
            <div class="error-message" id="errorMessage"></div>
            <div class="sample-merchants" id="sampleMerchants"><h4>Sample Merchants:</h4><div id="merchantsList"></div></div>
        </div>
        <div class="chat-container" id="chatContainer">
            <div class="messages" id="messages"></div>
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Ask about your transactions..." />
                <button id="sendButton">Send</button>
            </div>
        </div>
    </div>
    <script>
        var ws, sessionToken, currentMessage = '';
        
        document.getElementById('userType').onchange = function() {
            var c = this.value === 'customer';
            document.getElementById('identifierLabel').textContent = c ? 'VPA' : 'Merchant VPA';
            document.getElementById('pinLabel').textContent = c ? 'PIN' : 'Password';
            if (!c) {
                fetch('/api/sample-merchants').then(r=>r.json()).then(d=>{
                    var l = document.getElementById('merchantsList'); l.innerHTML = '';
                    (d.merchants||[]).forEach(m=>{
                        var div = document.createElement('div'); div.className = 'merchant-item';
                        div.textContent = m.vpa + ' - ' + m.name + ' (Pass: ' + m.password + ')';
                        div.onclick = ()=>{ document.getElementById('identifier').value = m.vpa; document.getElementById('pinOrPassword').value = m.password; };
                        l.appendChild(div);
                    });
                    document.getElementById('sampleMerchants').classList.add('show');
                });
            } else document.getElementById('sampleMerchants').classList.remove('show');
        };
        
        document.getElementById('authButton').onclick = function() {
            var fd = new FormData();
            fd.append('vpa', document.getElementById('identifier').value);
            fd.append('pin_or_password', document.getElementById('pinOrPassword').value);
            fd.append('user_type', document.getElementById('userType').value);
            fd.append('auth_method', 'vpa');
            this.disabled = true; this.textContent = 'Authenticating...';
            fetch('/api/authenticate', {method:'POST',body:fd}).then(r=>r.json().then(d=>({s:r.status,d}))).then(r=>{
                if(r.s===200){ 
                    sessionToken=r.d.session_token; 
                    document.getElementById('userInfo').textContent='👤 '+r.d.user_name;
                    document.getElementById('authContainer').classList.add('hidden'); 
                    document.getElementById('chatContainer').classList.add('show');
                    connectWS();
                    addMsg('assistant','Welcome! How can I help you?');
                } else { 
                    document.getElementById('errorMessage').textContent=r.d.error; 
                    document.getElementById('errorMessage').classList.add('show');
                    document.getElementById('authButton').disabled=false; 
                    document.getElementById('authButton').textContent='🔑 Authenticate'; 
                }
            });
        };
        
        function connectWS() {
            var url = (location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws/chat/'+sessionToken;
            console.log('Connecting to:', url);
            ws = new WebSocket(url);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(e) {
                console.log('Received:', e.data);
                var d = JSON.parse(e.data);
                if(d.type === 'response') { 
                    currentMessage += d.chunk; 
                    updateLast(currentMessage); 
                } else if(d.type === 'complete') { 
                    console.log('Complete, total length:', currentMessage.length);
                    currentMessage = ''; 
                    document.getElementById('sendButton').disabled = false; 
                } else if(d.type === 'error') {
                    updateLast('Error: ' + d.message);
                    document.getElementById('sendButton').disabled = false;
                }
            };
            
            ws.onerror = function(e) {
                console.error('WebSocket error:', e);
            };
            
            ws.onclose = function() {
                console.log('WebSocket closed');
            };
        }
        
        document.getElementById('sendButton').onclick = send;
        document.getElementById('messageInput').onkeypress = e=>{ if(e.key==='Enter')send(); };
        
        function send() { 
            var m = document.getElementById('messageInput').value.trim(); 
            if(!m || !ws || ws.readyState !== WebSocket.OPEN) {
                console.log('Cannot send:', !m ? 'empty message' : 'ws not ready');
                return;
            }
            addMsg('user', m); 
            document.getElementById('messageInput').value = ''; 
            document.getElementById('sendButton').disabled = true;
            currentMessage = ''; 
            addMsg('assistant', '⏳ Processing...'); 
            console.log('Sending:', m);
            ws.send(JSON.stringify({message: m})); 
        }
        
        function addMsg(t, c) { 
            var d = document.createElement('div'); 
            d.className = 'message ' + t;
            d.innerHTML = '<div class="message-content"></div>'; 
            d.querySelector('.message-content').textContent = c;
            document.getElementById('messages').appendChild(d); 
            document.getElementById('messages').scrollTop = 999999; 
        }
        
        function updateLast(c) { 
            var l = document.getElementById('messages').lastElementChild;
            if(l) {
                l.querySelector('.message-content').textContent = c; 
                document.getElementById('messages').scrollTop = 999999;
            }
        }
    </script>
</body>
</html>''')


if __name__ == "__main__":
    print("🚀 Starting Web UI Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)