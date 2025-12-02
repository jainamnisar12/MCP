import os
import sys
import time
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

# Add parent directory to path to import customer_auth and config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from customer_auth import CustomerAuthenticator
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from google.genai import types

# --- Performance Metrics Classes ---
perf_logger = logging.getLogger('agent_performance')
perf_logger.setLevel(logging.INFO)

# Determine absolute path for logs (use project root, not agent subdirectory)
log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
perf_log_path = os.path.join(log_dir, 'agent_performance.log')

perf_handler = logging.FileHandler(perf_log_path)
perf_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
perf_logger.addHandler(perf_handler)

@dataclass
class PerformanceMetrics:
    """Store performance metrics for a single request"""
    query_id: str  # Unique identifier to correlate with MCP tool logs
    timestamp: str
    user: str
    user_type: str
    query: str

    # End-to-end timing metrics (in seconds)
    total_conversation_time: float = 0.0  # Complete time from user input to final response
    agent_processing_time: Optional[float] = None  # Agent internal processing time
    tool_execution_time: Optional[float] = None  # Time spent executing tools (MCP)
    
    # Additional metrics
    tools_used: list = None  # List of tools that were called
    response_tokens: Optional[int] = None  # Number of tokens in response

    # Status
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class PerformanceTracker:
    """Track and aggregate performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.session_stats = defaultdict(lambda: {
            'total_queries': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'errors': 0
        })
    
    def log_metric(self, metric: PerformanceMetrics):
        """Log a performance metric"""
        self.metrics_history.append(metric)
        
        # Update session stats
        user_stats = self.session_stats[metric.user]
        user_stats['total_queries'] += 1
        user_stats['total_time'] += metric.total_conversation_time
        user_stats['avg_time'] = user_stats['total_time'] / user_stats['total_queries']
        
        if metric.status == "ERROR":
            user_stats['errors'] += 1
        
        # Log to file
        perf_logger.info(metric.to_json())
    
    def get_session_summary(self, user: str) -> Dict[str, Any]:
        """Get summary statistics for a user session"""
        stats = self.session_stats[user]
        
        # Calculate additional metrics from history
        user_metrics = [m for m in self.metrics_history if m.user == user]
        
        if user_metrics:
            stats['min_response_time'] = min(m.total_conversation_time for m in user_metrics)
            stats['max_response_time'] = max(m.total_conversation_time for m in user_metrics)
        
        return stats
    
    def print_summary(self, user: str):
        """Print formatted session summary"""
        stats = self.get_session_summary(user)
        
        print("\n" + "="*60)
        print("📊 SESSION PERFORMANCE SUMMARY")
        print("="*60)
        print(f"User: {user}")
        print(f"\n📈 Query Statistics:")
        print(f"   • Total Queries: {stats['total_queries']}")
        print(f"   • Errors: {stats['errors']}")
        
        print(f"\n⏱️  Response Times:")
        print(f"   • Average: {stats['avg_time']:.3f}s")
        if 'min_response_time' in stats:
            print(f"   • Fastest: {stats['min_response_time']:.3f}s")
            print(f"   • Slowest: {stats['max_response_time']:.3f}s")
        
        print(f"\n⏰ Total Session Time: {stats['total_time']:.3f}s")
        print("="*60)

# Global performance tracker instance
performance_tracker = PerformanceTracker()

# --- Authenticate user at startup (customer or merchant) ---
authenticator = CustomerAuthenticator()
user_data, user_type = authenticator.get_authenticated_user()

if not user_data or not user_type:
    print("\n❌ Exiting due to authentication failure.")
    exit(1)

# Set user variables based on user type
if user_type == 'customer':
    CURRENT_USER = user_data['name']
    CURRENT_USER_VPA = user_data['primary_vpa']
    CUSTOMER_ID = user_data.get('customer_id')
    USER_TYPE = 'customer'
elif user_type == 'merchant':
    CURRENT_USER = user_data['merchant_vpa']  # For merchants, use VPA as identifier
    CURRENT_USER_VPA = user_data['merchant_vpa']
    MERCHANT_ID = user_data.get('merchant_id')
    MERCHANT_NAME = user_data.get('merchant_name')
    USER_TYPE = 'merchant'
else:
    print(f"\n❌ Unknown user type: {user_type}")
    exit(1)

# --- Verify GCP Configuration ---
if not hasattr(config, 'GCP_PROJECT_ID') or not config.GCP_PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID not found in config. Please add it.")

if not hasattr(config, 'GCP_LOCATION'):
    config.GCP_LOCATION = "us-central1"
    print(f"⚠️  GCP_LOCATION not set in config. Using default: {config.GCP_LOCATION}")

print(f"\n{'='*60}")
print(f"🔧 VERTEX AI CONFIGURATION")
print(f"{'='*60}")
print(f"Project ID: {config.GCP_PROJECT_ID}")
print(f"Location: {config.GCP_LOCATION}")
print(f"Model: gemini-2.5-flash")
print(f"{'='*60}\n")

# --- Define the connection to your MCP server ---
mcp_tools = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:8001/sse"
    )
)

# --- Build user-specific instruction based on user type ---
if USER_TYPE == 'customer':
    user_intro = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔐 AUTHENTICATED CUSTOMER: {CURRENT_USER}\n"
        f"📱 VPA: {CURRENT_USER_VPA}\n"
        f"🔒 SECURITY LEVEL: MAXIMUM (Banking Grade)\n"
        f"🛡️ ACCESS SCOPE: Your Personal Data Only\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    user_context = f"Customer can ONLY access their own data:\n   → Politely explain: 'For security reasons, you can only access your own banking data. You are logged in as {CURRENT_USER} ({CURRENT_USER_VPA}).'"
elif USER_TYPE == 'merchant':
    user_intro = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔐 AUTHENTICATED MERCHANT: {MERCHANT_NAME}\n"
        f"📱 VPA: {CURRENT_USER_VPA}\n"
        f"🏪 MERCHANT ID: {MERCHANT_ID}\n"
        f"🔒 SECURITY LEVEL: MAXIMUM (Banking Grade)\n"
        f"🛡️ ACCESS SCOPE: All Transactions to Your Store\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    user_context = f"Merchant can access ALL transactions to their store:\n   → You are logged in as {MERCHANT_NAME} ({CURRENT_USER_VPA}). You can view all incoming payments to your store."

# --- Define the Main Agent with Vertex AI ---
# UPDATE ONLY THE AGENT INSTRUCTION in your existing agent.py
# Replace the instruction= part of root_agent with this:

root_agent = Agent(
    name="secure_banking_agent",
    model=Gemini(
        model_name="gemini-2.5-flash",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
        thinking_budget=0,
    ),
    instruction=(
        f"You are a friendly and secure banking assistant with access to specialized tools.\n"
        f"\n"
        f"{user_intro}"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"⚠️ CRITICAL OUTPUT RULES - READ CAREFULLY ⚠️\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"NEVER SHOW TO USER:\n"
        f"❌ JSON responses from tools (no curly braces {{}}, no square brackets [])\n"
        f"❌ Tool metadata (status, execution_time, row_count, generation_time, etc.)\n"
        f"❌ Raw data arrays or quotes around data\n"
        f"❌ Duplicate transaction lists - SHOW RESULTS ONLY ONCE\n"
        f"\n"
        f"ALWAYS SHOW TO USER:\n"
        f"✅ SQL query in a ```sql code block\n"
        f"✅ Tool results displayed exactly as received (already formatted)\n"
        f"✅ Simple, readable text only\n"
        f"\n"
        f"⚠️ ANTI-DUPLICATION RULE:\n"
        f"The execute_sql_query tool returns pre-formatted results with emojis.\n"
        f"DO NOT repeat, summarize, or reformat these results.\n"
        f"DO NOT say 'Here are the transactions:' and list them again.\n"
        f"The tool output IS the final answer - display it ONCE and stop.\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"DATABASE QUERY WORKFLOW (EXACT STEPS)\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"STEP 1: Call generate_sql_for_query\n"
        f"   → Tool returns: SQL query string (raw SQL, no code block)\n"
        f"   → YOU must wrap it in ```sql ... ``` code block when displaying\n"
        f"\n"
        f"STEP 2: Call execute_sql_query\n"
        f"   → Tool returns: Formatted transaction list (no header/footer)\n"
        f"   → Display results exactly as received - DO NOT MODIFY\n"
        f"\n"
        f"STEP 3: STOP - Do not add anything else\n"
        f"   → No summary, no repeat, no additional formatting\n"
        f"\n"
        f"CORRECT OUTPUT FORMAT:\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"```sql\n"
        f"SELECT * FROM transactions WHERE payee_vpa = 'merchant@bank'\n"
        f"```\n"
        f"\n"
        f"📊 Found 15 transaction(s)\n"
        f"⏱️ Retrieved in 1.312s\n"
        f"...[rest of formatted results from tool]...\n"
        f"✅ Total: 15 transaction(s)\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"\n"
        f"WRONG OUTPUT (DO NOT DO THIS):\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"```sql\n"
        f"SELECT * FROM transactions...\n"
        f"```\n"
        f"📊 Found 15 transaction(s)...\n"
        f"✅ Total: 15 transaction(s)\n"
        f"\n"
        f"Here are your transactions:        ← ❌ WRONG - DUPLICATE!\n"
        f"1. Transaction #1...               ← ❌ WRONG - DUPLICATE!\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"FOR PDF/DOCUMENT QUERIES\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"1. The ask_upi_document tool returns the answer text ready to display\n"
        f"2. Display the answer directly to the user - no modifications needed\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"AVAILABLE TOOLS\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"1. ask_upi_document\n"
        f"   Purpose: Answer questions about UPI (Unified Payments Interface)\n"
        f"   Use for: UPI process, features, security, limits, history\n"
        f"   Example: 'How does UPI work?', 'What are UPI transaction limits?'\n"
        f"\n"
        f"2. generate_sql_for_query\n"
        f"   Purpose: Generate SQL from natural language\n"
        f"   Security: Multi-layer validation, READ-ONLY access\n"
        f"   Use: generate_sql_for_query(natural_language_query='...', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"   Returns: SQL query string directly (NOT JSON)\n"
        f"\n"
        f"3. execute_sql_query\n"
        f"   Purpose: Execute validated SQL and return results\n"
        f"   Security: Row-level security enforced\n"
        f"   Use: execute_sql_query(sql_query='...', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"   Returns: Formatted results string directly (NOT JSON)\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"CRITICAL: DATABASE QUERY SECURITY PROTOCOL\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"When calling database tools, you MUST follow these rules:\n"
        f"\n"
        f"1. ✓ ALWAYS pass THREE required parameters:\n"
        f"   \n"
        f"   generate_sql_for_query(\n"
        f"       natural_language_query='[user question with context]',\n"
        f"       current_user='{CURRENT_USER}',\n"
        f"       user_type='{USER_TYPE}'\n"
        f"   )\n"
        f"\n"
        f"2. ✓ Convert user pronouns to explicit user name:\n"
        f"   \n"
        f"   User says: 'show my transactions'\n"
        f"   You call with: 'show transactions for {CURRENT_USER}'\n"
        f"   \n"
        f"   User says: 'what's my balance?'\n"
        f"   You call with: 'account balance for {CURRENT_USER}'\n"
        f"   \n"
        f"   User says: 'how much did I spend?'\n"
        f"   You call with: 'total spending for {CURRENT_USER}'\n"
        f"\n"
        f"3. ✓ NEVER omit the current_user or user_type parameters:\n"
        f"   \n"
        f"   ✗ WRONG: generate_sql_for_query('show transactions')\n"
        f"   ✓ RIGHT: generate_sql_for_query('show transactions for {CURRENT_USER}', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"\n"
        f"4. ✓ Maintain context in follow-up queries:\n"
        f"   \n"
        f"   First query: 'show my transactions'\n"
        f"   → generate_sql_for_query('show transactions for {CURRENT_USER}', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"   \n"
        f"   Follow-up: 'what's the average?'\n"
        f"   → generate_sql_for_query('average transaction amount for {CURRENT_USER}', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"\n"
        f"5. ✓ Make queries self-contained:\n"
        f"   \n"
        f"   Each query should be complete and include the user's name, even in conversations.\n"
        f"   Don't rely on previous queries - the database tool doesn't have conversation memory.\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"TWO-STEP DATABASE QUERY WORKFLOW (MANDATORY)\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"For ANY database query, you MUST follow this exact two-step process:\n"
        f"\n"
        f"STEP 1: Generate SQL\n"
        f"───────────────────\n"
        f"Call: generate_sql_for_query(\n"
        f"    natural_language_query='show transactions for {CURRENT_USER}',\n"
        f"    current_user='{CURRENT_USER}',\n"
        f"    user_type='{USER_TYPE}'\n"
        f")\n"
        f"\n"
        f"Tool returns: SQL query string directly\n"
        f"\n"
        f"Show the SQL:\n"
        f"```sql\n"
        f"[SQL_QUERY_HERE]\n"
        f"```\n"
        f"\n"
        f"STEP 2: Execute SQL\n"
        f"───────────────────\n"
        f"Call: execute_sql_query(\n"
        f"    sql_query='[SQL from step 1]',\n"
        f"    current_user='{CURRENT_USER}',\n"
        f"    user_type='{USER_TYPE}'\n"
        f")\n"
        f"\n"
        f"Tool returns: Formatted results string directly\n"
        f"\n"
        f"Display the results exactly as received - THEN STOP.\n"
        f"DO NOT add any text after the results.\n"
        f"DO NOT repeat or summarize the transactions.\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"SECURITY & ACCESS CONTROL\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"1. 🚫 Unauthorized Access Attempts:\n"
        f"   \n"
        f"   {user_context}\n"
        f"   \n"
        f"   Examples of requests to DENY (for customers):\n"
        f"   • 'Show all customers' → DENY\n"
        f"   • 'What are other people's transactions?' → DENY\n"
        f"   • 'List all users' → DENY\n"
        f"\n"
        f"2. 🛡️ READ-ONLY Access:\n"
        f"   \n"
        f"   The database is READ-ONLY. You cannot:\n"
        f"   • Modify data (UPDATE)\n"
        f"   • Delete records (DELETE)\n"
        f"   • Add new records (INSERT)\n"
        f"   • Change database structure (ALTER, CREATE, DROP)\n"
        f"   \n"
        f"   If user requests modifications:\n"
        f"   → Explain: 'I have read-only access to the database for security reasons. I cannot modify, delete, or add records. Please contact your bank for account modifications.'\n"
        f"\n"
        f"3. ⚠️ Ambiguous Requests:\n"
        f"   \n"
        f"   If unsure about a query, ask for clarification rather than guessing.\n"
        f"   Better to confirm than to risk a security violation.\n"
        f"\n"
        f"4. 🔒 Rate Limiting:\n"
        f"   \n"
        f"   The system enforces rate limits:\n"
        f"   • 10 queries per minute\n"
        f"   • 100 queries per session\n"
        f"   \n"
        f"   If user hits limit:\n"
        f"   → Suggest: 'You've reached the query limit. Please wait a moment or start a new session.'\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"CONVERSATION BEST PRACTICES\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"1. 📝 Context Awareness:\n"
        f"   • Remember the full conversation history\n"
        f"   • Make each query self-contained with user name\n"
        f"   • Don't assume the tool remembers previous queries\n"
        f"\n"
        f"2. 💬 Conversational Tone:\n"
        f"   • Be friendly and helpful\n"
        f"   • Use clear, non-technical language\n"
        f"   • Explain financial terms when needed\n"
        f"\n"
        f"3. 📊 Data Presentation:\n"
        f"   • The tool already formats data with emojis and currency symbols\n"
        f"   • Just display tool results as-is - no reformatting needed\n"
        f"   • DO NOT duplicate or repeat the transaction list\n"
        f"\n"
        f"4. 🎯 Accuracy:\n"
        f"   • Always verify you're using the correct tool\n"
        f"   • Don't make assumptions about data\n"
        f"   • If data is missing, say so clearly\n"
        f"   • Don't invent or estimate values\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"REMEMBER: Security is paramount. When in doubt, always:\n"
        f"1. Include BOTH current_user AND user_type parameters\n"
        f"2. Make queries explicit with the user's name/VPA\n"
        f"3. Show SQL query in code block, then tool results as-is\n"
        f"4. DO NOT DUPLICATE - show results only ONCE\n"
        f"5. Protect user privacy and data integrity\n"
        f"6. Use the two-step workflow for ALL database queries\n"
        f"7. Current user: {CURRENT_USER} | Type: {USER_TYPE}\n"
        f"═══════════════════════════════════════════════════════════════════\n"
    ),
    tools=[mcp_tools]
)

# --- Create Runner ---
session_service = InMemorySessionService()
runner = Runner(
    app_name="agent",
    agent=root_agent,
    session_service=session_service
)

# Generate a session ID for this user
import uuid
USER_SESSION_ID = str(uuid.uuid4())

# --- Main execution block ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"✓ Secure Banking Assistant Ready")
    print("=" * 60)

    if USER_TYPE == 'customer':
        print(f"👤 Customer: {CURRENT_USER}")
        print(f"📱 VPA: {CURRENT_USER_VPA}")
        print(f"🆔 Customer ID: {CUSTOMER_ID}")
        print(f"🛡️ Access Scope: Your Personal Data Only")
    elif USER_TYPE == 'merchant':
        print(f"🏪 Merchant: {MERCHANT_NAME}")
        print(f"📱 VPA: {CURRENT_USER_VPA}")
        print(f"🆔 Merchant ID: {MERCHANT_ID}")
        print(f"🛡️ Access Scope: All Transactions to Your Store")

    print(f"🔒 Security Level: Banking Grade (Multi-Layer)")
    print(f"🌐 AI Platform: Vertex AI ({config.GCP_PROJECT_ID})")
    print(f"🔧 Model: gemini-2.5-flash")
    print(f"👥 User Type: {USER_TYPE.upper()}")
    print("\n📋 Security Features Active:")

    if USER_TYPE == 'customer':
        print("   ✓ VPA + PIN Authentication")
        print("   ✓ Query Parser & Validator")
        print("   ✓ Row-Level Security (Your Data Only)")
    elif USER_TYPE == 'merchant':
        print("   ✓ VPA + Password Authentication")
        print("   ✓ Query Parser & Validator")
        print("   ✓ Access to Store Transactions")

    print("   ✓ Rate Limiting (10/min, 100/session)")
    print("   ✓ READ-ONLY Database Access")
    print("   ✓ Comprehensive Audit Logging")
    print("\n🚫 Prohibited Operations:")
    print("   • DELETE, UPDATE, INSERT")
    print("   • Schema modifications")

    if USER_TYPE == 'customer':
        print("   • Access to other customers' data")
        print("\n✅ Allowed Operations:")
        print("   • Query your own transactions")
        print("   • View your account details")
    elif USER_TYPE == 'merchant':
        print("   • Access to customer personal information")
        print("\n✅ Allowed Operations:")
        print("   • Query all transactions to your store")
        print("   • View sales statistics and analytics")

    print("   • Ask UPI-related questions")
    
    print("\n📊 Performance Monitoring: ENABLED")
    print("   • Metrics logged to: agent_performance.log")
    print("   • Real-time timing displayed per query")
    
    print("=" * 60)
    print("\nType 'quit', 'exit', or 'q' to stop.")
    print("All queries are logged for security and compliance.\n")
    
    # Interactive loop with performance tracking
    conversation_count = 0
    session_start_time = time.time()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                session_duration = time.time() - session_start_time
                
                print("\n" + "=" * 60)
                print("👋 Thank you for using Secure Banking Assistant")
                print("=" * 60)
                print(f"📊 Session Summary:")
                print(f"   • Total interactions: {conversation_count}")
                print(f"   • Session duration: {session_duration:.2f}s")
                print(f"   • User: {CURRENT_USER}")
                print(f"   • All queries logged for audit purposes")
                print("=" * 60)
                
                # Print detailed performance summary
                if conversation_count > 0:
                    performance_tracker.print_summary(CURRENT_USER)
                
                print("\n🔒 Your session has been securely closed.\n")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            conversation_count += 1
            query_number = conversation_count
            
            print(f"\n{'─'*60}")
            print(f"Query #{query_number}")
            print(f"{'─'*60}")
            print("Assistant: ", end="", flush=True)
            
            # Start performance tracking
            conversation_start_time = time.time()
            query_id = str(uuid.uuid4())[:8]  # Short unique ID for correlation
            
            metric = PerformanceMetrics(
                query_id=query_id,
                timestamp=datetime.now().isoformat(),
                user=CURRENT_USER,
                user_type=USER_TYPE,
                query=user_input[:200],  # Truncate long queries for logging
                tools_used=[]
            )
            
            print(f"🔍 Query ID: {query_id}")  # Display for correlation
            
            # Run the agent with the user's query
            try:
                agent_start = time.time()

                # Helper to collect async generator results
                async def collect_response():
                    chunks = []
                    tool_calls_detected = False
                    async for event in runner.run_async(
                        user_id=CURRENT_USER,
                        session_id=USER_SESSION_ID,
                        new_message=Content(parts=[Part(text=user_input)])  # ✅ NEW - CORRECT
                    ):
                        # Extract text from events
                        if hasattr(event, 'content') and hasattr(event.content, 'text'):
                            chunk_str = event.content.text
                            chunks.append(chunk_str)
                            print(chunk_str, end='', flush=True)
                            
                        # Detect tool usage (this is approximate, actual tool detection would need deeper integration)
                        if hasattr(event, 'content'):
                            content_str = str(event.content).lower()
                            # Check for all available tools
                            tools_to_check = [
                                'query_customer_database', 
                                'query_pdf_documents',
                                'generate_sql_for_query',
                                'execute_sql_query',
                                'ask_upi_document'
                            ]
                            
                            for tool_name in tools_to_check:
                                if tool_name in content_str and tool_name not in metric.tools_used:
                                    metric.tools_used.append(tool_name)
                    
                    print()  # New line after response
                    return "".join(chunks) if chunks else "(No response)"

                # Run the async function
                import asyncio
                response = asyncio.run(collect_response())

                metric.agent_processing_time = time.time() - agent_start
                
                # Calculate total conversation time (end-to-end)
                metric.total_conversation_time = time.time() - conversation_start_time
                metric.status = "SUCCESS"
                
                # Estimate response tokens (approximate)
                metric.response_tokens = len(response.split()) if response else 0
                
                # Log the metric
                performance_tracker.log_metric(metric)
                
                # Print performance summary for this query
                print(f"\n{'─'*60}")
                print(f"⏱️  Performance Metrics (Query #{query_number}):")
                print(f"   • End-to-End Time: {metric.total_conversation_time:.3f}s")
                print(f"   • Agent Processing: {metric.agent_processing_time:.3f}s")
                if metric.tools_used:
                    print(f"   • Tools Used: {', '.join(metric.tools_used)}")
                if metric.response_tokens:
                    print(f"   • Response Tokens: {metric.response_tokens}")
                
                # Get current session stats
                stats = performance_tracker.get_session_summary(CURRENT_USER)
                print(f"   • Session Average: {stats['avg_time']:.3f}s")
                print(f"{'─'*60}\n")
                
            except Exception as agent_error:
                metric.total_conversation_time = time.time() - conversation_start_time
                metric.status = "ERROR"
                metric.error_message = str(agent_error)[:200]
                
                performance_tracker.log_metric(metric)
                
                print(f"\n⚠️ I encountered an issue processing your request.")
                print(f"Error details: {str(agent_error)}")
                
                print(f"\n{'─'*60}")
                print(f"⏱️  Time taken: {metric.total_time:.3f}s (ERROR)")
                print(f"{'─'*60}")
                print("Please try rephrasing your question or contact support if the issue persists.\n")
            
        except KeyboardInterrupt:
            session_duration = time.time() - session_start_time
            
            print("\n\n" + "=" * 60)
            print("👋 Session interrupted by user")
            print("=" * 60)
            print(f"📊 Session Summary:")
            print(f"   • Total interactions: {conversation_count}")
            print(f"   • Session duration: {session_duration:.2f}s")
            print(f"   • User: {CURRENT_USER}")
            print("=" * 60)
            
            # Print detailed performance summary
            if conversation_count > 0:
                performance_tracker.print_summary(CURRENT_USER)
            
            print("\n🔒 Your session has been securely closed.\n")
            break
            
        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}")
            print("Please try again or type 'quit' to exit.\n")