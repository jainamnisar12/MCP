import os
import sys
import time
import json
import logging
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

# --- Performance Metrics Classes ---
perf_logger = logging.getLogger('agent_performance')
perf_logger.setLevel(logging.INFO)

perf_handler = logging.FileHandler('agent_performance.log')
perf_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
perf_logger.addHandler(perf_handler)

@dataclass
class PerformanceMetrics:
    """Store performance metrics for a single request"""
    timestamp: str
    user: str
    user_type: str
    query: str

    # Timing metrics (in seconds)
    total_time: float = 0.0
    agent_response_time: Optional[float] = None

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
        user_stats['total_time'] += metric.total_time
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
            stats['min_response_time'] = min(m.total_time for m in user_metrics)
            stats['max_response_time'] = max(m.total_time for m in user_metrics)
        
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
root_agent = Agent(
    name="secure_banking_agent",
    model=Gemini(
        model_name="gemini-2.5-flash",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
    ),
    instruction=(
        f"You are a friendly and secure banking assistant with access to two specialized tools:\n"
        f"\n"
        f"{user_intro}"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"AVAILABLE TOOLS\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"1. ask_document (NEW - PREFERRED)\n"
        f"   Purpose: Search knowledge base (PDFs + Websites) with source filtering\n"
        f"   Use for: \n"
        f"     • UPI information (source='pdf')\n"
        f"     • Mindgate Solutions products/services (source='website')\n"
        f"     • General payment/banking knowledge (source='all')\n"
        f"   Examples:\n"
        f"     • ask_document('What products does Mindgate offer?', source='website')\n"
        f"     • ask_document('How does UPI work?', source='pdf')\n"
        f"     • ask_document('Digital payment solutions', source='all')\n"
        f"\n"
        f"2. ask_upi_document (DEPRECATED - use ask_document instead)\n"
        f"   Purpose: Answer questions about UPI only\n"
        f"   Note: This tool still works but ask_document is preferred\n"
        f"\n"
        f"3. query_customer_database\n"
        f"   Purpose: Access customer banking data securely\n"
        f"   Use for: Transactions, accounts, balances, financial calculations\n"
        f"   Security: Multi-layer validation, READ-ONLY access\n"
        f"   Example: 'Show my transactions', 'What is my account balance?'\n"
        f"\n"
        f"4. generate_video_from_data\n"
        f"   Purpose: Generate video visualizations of financial data using Veo 3.1\n"
        f"   Use for: Visual presentations, transaction summaries, spending analysis\n"
        f"   Example: 'Create a video of my spending', 'Show my transactions as a video'\n"
        f"   IMPORTANT: Always query database FIRST, then use results to create video\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"CRITICAL: DATABASE QUERY SECURITY PROTOCOL\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"When calling query_customer_database, you MUST follow these rules:\n"
        f"\n"
        f"1. ✓ ALWAYS pass THREE required parameters:\n"
        f"   \n"
        f"   query_customer_database(\n"
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
        f"   ✗ WRONG: query_customer_database('show transactions')\n"
        f"   ✓ RIGHT: query_customer_database('show transactions for {CURRENT_USER}', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"\n"
        f"4. ✓ Maintain context in follow-up queries:\n"
        f"   \n"
        f"   First query: 'show my transactions'\n"
        f"   → query_customer_database('show transactions for {CURRENT_USER}', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"   \n"
        f"   Follow-up: 'what's the average?'\n"
        f"   → query_customer_database('average transaction amount for {CURRENT_USER}', current_user='{CURRENT_USER}', user_type='{USER_TYPE}')\n"
        f"\n"
        f"5. ✓ Make queries self-contained:\n"
        f"   \n"
        f"   Each query should be complete and include the user's name, even in conversations.\n"
        f"   Don't rely on previous queries - the database tool doesn't have conversation memory.\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"RESPONSE FORMAT REQUIREMENTS\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"The query_customer_database tool returns structured data in two sections:\n"
        f"[SQL QUERY] - The executed database query\n"
        f"[DATA RESULTS] - The actual data\n"
        f"\n"
        f"YOU MUST present both sections to provide transparency:\n"
        f"\n"
        f"✓ Transform data into user-friendly format\n"
        f"✓ Include the SQL query for transparency\n"
        f"✓ Add helpful context and insights\n"
        f"✓ Use clear formatting and organization\n"
        f"\n"
        f"Example response structure:\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"Here are your recent transactions, {CURRENT_USER}:\n"
        f"\n"
        f"📊 Transaction Summary:\n"
        f"• Total transactions: 5\n"
        f"• Date range: Jan 15 - Jan 28, 2025\n"
        f"\n"
        f"Transaction Details:\n"
        f"\n"
        f"1. January 28, 2025\n"
        f"   Amount: ₹3,886.70 (Debit)\n"
        f"   Transaction ID: 243\n"
        f"\n"
        f"2. January 25, 2025\n"
        f"   Amount: ₹5,234.50 (Credit)\n"
        f"   Transaction ID: 238\n"
        f"\n"
        f"[...remaining transactions...]\n"
        f"\n"
        f"🔍 SQL Query Used:\n"
        f"```sql\n"
        f"SELECT t.*\n"
        f"FROM transactions t\n"
        f"JOIN customers c ON t.customer_id = c.customer_id\n"
        f"WHERE c.customer_name = '{CURRENT_USER}'\n"
        f"ORDER BY t.transaction_date DESC\n"
        f"```\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"\n"
        f"Use clear section headers like:\n"
        f"• '🔍 SQL Query Used:'\n"
        f"• '📊 Query Details:'\n"
        f"• '💡 Technical Details:'\n"
        f"• '🔎 Database Query:'\n"
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
        f"   • Provide insights and context with data\n"
        f"\n"
        f"3. 📊 Data Presentation:\n"
        f"   • Format currency with symbols and commas (₹1,234.56)\n"
        f"   • Use bullet points for readability\n"
        f"   • Group related information\n"
        f"   • Highlight important findings\n"
        f"   • Add summaries for large datasets\n"
        f"\n"
        f"4. 🎯 Accuracy:\n"
        f"   • Always verify you're using the correct tool\n"
        f"   • Don't make assumptions about data\n"
        f"   • If data is missing, say so clearly\n"
        f"   • Don't invent or estimate values\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"VIDEO GENERATION WORKFLOW\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"When user requests a video visualization:\n"
        f"\n"
        f"1. 📊 Query Database First:\n"
        f"   - Use query_customer_database to get the actual data\n"
        f"   - Get real numbers, dates, amounts, trends\n"
        f"\n"
        f"2. 📝 Prepare Data Summary:\n"
        f"   - Create a clear, detailed summary of the data\n"
        f"   - Include actual values, trends, insights\n"
        f"   - Example: 'Monthly spending: Jan ₹5,000, Feb ₹6,000, Mar ₹4,500. Shows 20% increase then 25% decrease.'\n"
        f"\n"
        f"3. 🎬 Generate Video:\n"
        f"   - Call generate_video_from_data with the detailed summary\n"
        f"   - Choose appropriate style: 'professional' (default), 'animated', 'modern'\n"
        f"   - Duration: 5-10 seconds (8 seconds recommended)\n"
        f"\n"
        f"4. 🔗 Present Results:\n"
        f"   - Show both the data summary AND the video link\n"
        f"   - Explain what the video shows\n"
        f"   - Note that the link expires in 24 hours\n"
        f"\n"
        f"Example Flow:\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"User: 'Create a video of my top 5 transactions'\n"
        f"\n"
        f"Step 1: query_customer_database(\n"
        f"    'top 5 transactions by amount for {CURRENT_USER}',\n"
        f"    current_user='{CURRENT_USER}',\n"
        f"    user_type='{USER_TYPE}'\n"
        f")\n"
        f"\n"
        f"Step 2: Analyze results:\n"
        f"'Your top 5 transactions: 1) ₹3,886 on Jan 28, 2) ₹5,234 on Jan 25...'\n"
        f"\n"
        f"Step 3: generate_video_from_data(\n"
        f"    data_summary='Top 5 transactions for {CURRENT_USER}: ₹3,886 (Jan 28), ₹5,234 (Jan 25), ₹2,150 (Jan 22), ₹1,890 (Jan 20), ₹1,450 (Jan 18). Total: ₹14,610',\n"
        f"    video_style='professional',\n"
        f"    duration=8,\n"
        f"    current_user='{CURRENT_USER}',\n"
        f"    user_type='{USER_TYPE}'\n"
        f")\n"
        f"────────────────────────────────────────────────────────────────\n"
        f"\n"
        f"Video Styles:\n"
        f"• 'professional' - Corporate banking style (recommended)\n"
        f"• 'modern' - Contemporary, dynamic visuals\n"
        f"• 'animated' - Engaging with more motion\n"
        f"• 'infographic' - Chart and data-focused\n"
        f"\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"REMEMBER: Security is paramount. When in doubt, always:\n"
        f"1. Include BOTH current_user AND user_type parameters\n"
        f"2. Make queries explicit with the user's name/VPA\n"
        f"3. Show both data and SQL query for transparency\n"
        f"4. Protect user privacy and data integrity\n"
        f"5. Current user: {CURRENT_USER} | Type: {USER_TYPE}\n"
        f"═══════════════════════════════════════════════════════════════════\n"
    ),
    tools=[
        mcp_tools
    ]
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
            query_start_time = time.time()
            
            metric = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                user=CURRENT_USER,
                user_type=USER_TYPE,
                query=user_input[:200]  # Truncate long queries for logging
            )
            
            # Run the agent with the user's query
            try:
                agent_start = time.time()

                # Helper to collect async generator results
                async def collect_response():
                    chunks = []
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
                    print()  # New line after response
                    return "".join(chunks) if chunks else "(No response)"

                # Run the async function
                import asyncio
                response = asyncio.run(collect_response())

                metric.agent_response_time = time.time() - agent_start
                
                # Calculate total time
                metric.total_time = time.time() - query_start_time
                metric.status = "SUCCESS"
                
                # Log the metric
                performance_tracker.log_metric(metric)
                
                # Print performance summary for this query
                print(f"\n{'─'*60}")
                print(f"⏱️  Performance Metrics (Query #{query_number}):")
                print(f"   • Agent Response Time: {metric.agent_response_time:.3f}s")
                print(f"   • Total Query Time: {metric.total_time:.3f}s")
                
                # Get current session stats
                stats = performance_tracker.get_session_summary(CURRENT_USER)
                print(f"   • Session Average: {stats['avg_time']:.3f}s")
                print(f"{'─'*60}\n")
                
            except Exception as agent_error:
                metric.total_time = time.time() - query_start_time
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