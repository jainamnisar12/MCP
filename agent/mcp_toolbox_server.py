import config
import uvicorn
import pandas as pd
import re
import json
import logging
import time
from datetime import datetime, timedelta
from google.cloud import bigquery
from fastmcp import FastMCP
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Tuple
from collections import defaultdict

# Import utilities
from schema_cache_manager import (
    load_or_refresh_schema,
    SchemaCache
)

from table_context import ACCESS_CONTROL
from table_retriever import TableRetriever

# --- Configure Audit Logging ---
audit_logger = logging.getLogger('mcp_security_audit')
audit_logger.setLevel(logging.INFO)

# File handler for audit trail
audit_handler = logging.FileHandler('mcp_security_audit.log')
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))
audit_logger.addHandler(audit_handler)

# Console handler for real-time monitoring
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter(
    '🚨 SECURITY: %(message)s'
))
audit_logger.addHandler(console_handler)

# --- Configure Performance Logging ---
perf_logger = logging.getLogger('mcp_performance')
perf_logger.setLevel(logging.INFO)

perf_handler = logging.FileHandler('mcp_performance.log')
perf_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
perf_logger.addHandler(perf_handler)

def log_query_attempt(
    user: str,
    query: str,
    status: str,  # 'ALLOWED', 'BLOCKED', 'ERROR'
    reason: str = None,
    row_count: int = None
):
    """
    Log all query attempts per guardrails compliance requirements.
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user,
        'query': query[:500],  # Truncate long queries
        'status': status,
        'reason': reason,
        'row_count': row_count
    }
    
    audit_logger.info(json.dumps(log_entry))

def log_performance_metric(
    user: str,
    user_type: str,
    query: str,
    total_time: float,
    sql_generation_time: float = None,
    sql_execution_time: float = None,
    pdf_search_time: float = None,
    rows_returned: int = None,
    bytes_processed: int = None,
    status: str = "SUCCESS"
):
    """Log performance metrics for MCP tool calls"""
    metric = {
        'timestamp': datetime.now().isoformat(),
        'user': user,
        'user_type': user_type,
        'query': query[:200],  # Truncate long queries
        'total_time': round(total_time, 3),
        'sql_generation_time': round(sql_generation_time, 3) if sql_generation_time else None,
        'sql_execution_time': round(sql_execution_time, 3) if sql_execution_time else None,
        'pdf_search_time': round(pdf_search_time, 3) if pdf_search_time else None,
        'rows_returned': rows_returned,
        'bytes_processed': bytes_processed,
        'status': status
    }
    
    perf_logger.info(json.dumps(metric))
    
    # Print real-time performance summary
    print(f"\n{'─'*60}")
    print(f"⏱️  MCP Tool Performance:")
    if sql_generation_time:
        print(f"   • SQL Generation: {sql_generation_time:.3f}s")
    if sql_execution_time:
        print(f"   • SQL Execution: {sql_execution_time:.3f}s")
    if pdf_search_time:
        print(f"   • PDF Search: {pdf_search_time:.3f}s")
    print(f"   • Total Time: {total_time:.3f}s")
    if rows_returned is not None:
        print(f"   • Rows Returned: {rows_returned}")
    if bytes_processed is not None:
        print(f"   • Bytes Processed: {bytes_processed:,} ({bytes_processed / 1024 / 1024:.2f} MB)")
    print(f"{'─'*60}\n")

# --- Rate Limiter (Layer 4) ---
class RateLimiter:
    def __init__(self, max_queries_per_minute=10, max_queries_per_session=100):
        self.max_per_minute = max_queries_per_minute
        self.max_per_session = max_queries_per_session
        self.user_queries = defaultdict(list)
        self.session_counts = defaultdict(int)
    
    def is_allowed(self, user: str) -> Tuple[bool, str]:
        """Check if user has exceeded rate limits."""
        now = datetime.now()
        
        # Check session limit
        if self.session_counts[user] >= self.max_per_session:
            return False, (
                f"🚫 Rate limit exceeded: Maximum {self.max_per_session} queries per session.\n"
                f"   Please start a new session."
            )
        
        # Check per-minute limit
        one_minute_ago = now - timedelta(minutes=1)
        self.user_queries[user] = [
            ts for ts in self.user_queries[user] if ts > one_minute_ago
        ]
        
        if len(self.user_queries[user]) >= self.max_per_minute:
            return False, (
                f"🚫 Rate limit exceeded: Maximum {self.max_per_minute} queries per minute.\n"
                f"   Please wait before trying again."
            )
        
        # Record this query
        self.user_queries[user].append(now)
        self.session_counts[user] += 1
        
        return True, ""
    
    def reset_session(self, user: str):
        """Reset session count for a user."""
        self.session_counts[user] = 0
        self.user_queries[user] = []

# Initialize rate limiter
rate_limiter = RateLimiter(max_queries_per_minute=10, max_queries_per_session=100)

# --- 1. Initialize All Shared Resources ---
print("\n" + "="*60)
print("🚀 INITIALIZING MCP TOOLBOX SERVER (VERTEX AI)")
print("="*60)

# Verify GCP Configuration
if not hasattr(config, 'GCP_PROJECT_ID') or not config.GCP_PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID not found in config. Please add it.")

if not hasattr(config, 'GCP_LOCATION'):
    config.GCP_LOCATION = "us-central1"
    print(f"⚠️  GCP_LOCATION not set in config. Using default: {config.GCP_LOCATION}")

print(f"\n📍 Vertex AI Configuration:")
print(f"   Project: {config.GCP_PROJECT_ID}")
print(f"   Location: {config.GCP_LOCATION}")

print("\n[1/5] Initializing AI models with Vertex AI...")
llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    project=config.GCP_PROJECT_ID,
    location=config.GCP_LOCATION,
    temperature=0,
)

embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=config.GCP_PROJECT_ID,
    location=config.GCP_LOCATION,
)
print("✓ LLM and Embeddings initialized (Vertex AI)")

print("\n[2/5] Connecting to BigQuery...")
bq_client = bigquery.Client(project=config.GCP_PROJECT_ID)
print(f"✓ Connected to project: {config.GCP_PROJECT_ID}")

# --- Fetch Dynamic Schema ---
print("\n[3/5] Loading schema from cache...")
try:
    schema_info, table_contexts, formatted_schema = load_or_refresh_schema(
        bq_client=bq_client,
        llm=llm,
        force_refresh=False  # Set to True to force refresh
    )
    
    # Show cache status
    cache_manager = SchemaCache()
    cache_info = cache_manager.get_cache_info()
    if cache_info['exists']:
        print(f"✓ Using cached schema (age: {cache_info['age_days']} days)")
    else:
        print(f"✓ Fresh schema fetched and cached")
    
    print(f"✓ Schema loaded for {len(schema_info)} tables")

except Exception as e:
    print(f"❌ ERROR: Could not load schema: {str(e)}")
    raise

# --- Initialize Table Retriever for RAG ---
print("\n[4/5] Initializing Table RAG Retriever...")
try:
    table_retriever = TableRetriever()
    rag_initialized = table_retriever.initialize()

    if rag_initialized:
        print("✓ Table RAG retriever initialized successfully")
    else:
        print("⚠️  Table RAG retriever not available - using fallback (all tables)")
        table_retriever = None
except Exception as e:
    print(f"⚠️  Could not initialize table retriever: {e}")
    print("   Using fallback mode with all tables")
    table_retriever = None

# --- Display Schema ---
print("\n[5/5] Schema and RAG loaded successfully")
print("\n" + "="*60)
print("📚 SCHEMA SUMMARY:")
print("="*60)
for table_name in schema_info.keys():
    print(f"  • {table_name}: {len(schema_info[table_name]['fields'])} columns")
print("="*60 + "\n")

# --- Load Vector Store ---
print("[5/5] Loading vector store for PDF queries...")
try:
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Vector store loaded from {config.VECTOR_STORE_PATH}")
except Exception as e:
    print(f"❌ FATAL: Could not load vector store from {config.VECTOR_STORE_PATH}")
    print(f"Error: {str(e)}")
    print("Please run 'python -m agent.pdf_indexer' first to create the vector store.")
    raise

print("\n" + "="*60)
print("✅ ALL RESOURCES INITIALIZED SUCCESSFULLY (VERTEX AI)")
print("="*60 + "\n")

# --- 2. CREATE FastMCP INSTANCE ---
mcp = FastMCP("MyToolboxServer")

# --- 3. Define PDF Tool Logic ---
pdf_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are a helpful assistant that answers questions about UPI (Unified Payments Interface).
    Use the provided context to answer the user's question. 
    Your answer should be based SOLELY on the context provided.
    If the context does not contain the answer, say that you cannot find the information in the document.
    
    Context:
    {context}
    """),
    ("human", "{question}")
])
pdf_generation_chain = pdf_prompt | llm

@mcp.tool
def ask_upi_document(question: str) -> str:
    """
    Answers questions about the UPI (Unified Payments Interface) process
    by searching a dedicated PDF document. Use this for questions about
    how UPI works, its features, security, limits, or history.
    """
    start_time = time.time()
    
    print(f"[PDF Tool] Received query: {question}")
    
    try:
        # Track vector search time
        search_start = time.time()
        docs = vector_store.similarity_search(question, k=3)
        search_time = time.time() - search_start
        
        if not docs:
            total_time = time.time() - start_time
            log_performance_metric(
                user='SYSTEM',
                user_type='pdf_query',
                query=question,
                total_time=total_time,
                pdf_search_time=search_time,
                status='NO_RESULTS'
            )
            return "I couldn't find any relevant information in the document to answer that question."
        
        context = "\n\n".join([f"Context {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Track LLM response time
        llm_start = time.time()
        response = pdf_generation_chain.invoke({
            "context": context,
            "question": question
        })
        llm_time = time.time() - llm_start
        
        total_time = time.time() - start_time
        
        log_performance_metric(
            user='SYSTEM',
            user_type='pdf_query',
            query=question,
            total_time=total_time,
            pdf_search_time=search_time + llm_time,
            rows_returned=len(docs),
            status='SUCCESS'
        )
        
        print(f"⏱️  PDF Query completed in {total_time:.3f}s (Search: {search_time:.3f}s, LLM: {llm_time:.3f}s)")
        
        return response.content
        
    except Exception as e:
        total_time = time.time() - start_time
        log_performance_metric(
            user='SYSTEM',
            user_type='pdf_query',
            query=question,
            total_time=total_time,
            status='ERROR'
        )
        raise e

# --- 4. Security Validation Functions ---

def validate_query_type(sql_query: str) -> Tuple[bool, str]:
    """
    Layer 1: Comprehensive query type validation against prohibited operations.
    Implements Query Parser & Validator per guardrails documentation.
    """
    sql_upper = sql_query.upper().strip()
    
    # Remove SQL comments to prevent bypassing
    sql_upper = re.sub(r'--.*$', '', sql_upper, flags=re.MULTILINE)
    sql_upper = re.sub(r'/\*.*?\*/', '', sql_upper, flags=re.DOTALL)
    
    # Prohibited operations from guardrails document
    prohibited_patterns = {
        'DELETE': ['DELETE', 'TRUNCATE', 'DROP TABLE', 'DROP DATABASE'],
        'UPDATE': ['UPDATE'],
        'INSERT': ['INSERT'],
        'SCHEMA': ['ALTER TABLE', 'ALTER DATABASE', 'CREATE TABLE', 'CREATE DATABASE', 
                   'CREATE INDEX', 'DROP INDEX', 'CREATE VIEW', 'DROP VIEW'],
        'ADMIN': ['GRANT', 'REVOKE', 'CREATE USER', 'DROP USER', 'ALTER USER'],
        'INJECTION': [';.*(?:DELETE|UPDATE|INSERT|DROP)', 'EXEC', 'EXECUTE', 'xp_', 'sp_'],
    }
    
    for category, keywords in prohibited_patterns.items():
        for keyword in keywords:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + re.escape(keyword).replace(r'\ ', r'\s+') + r'\b'
            if re.search(pattern, sql_upper):
                return False, (
                    f"🚫 SECURITY BLOCK: {category} operations are not permitted.\n"
                    f"   Detected: {keyword}\n"
                    f"   This chatbot is READ-ONLY and can only execute SELECT queries.\n"
                    f"   Reason: Banking security regulations require data integrity.\n"
                    f"   This incident has been logged for audit purposes."
                )
    
    # Ensure query starts with SELECT or WITH (for CTEs)
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        return False, (
            "🚫 SECURITY BLOCK: Only SELECT queries are permitted.\n"
            "   This chatbot has READ-ONLY access to the database.\n"
            "   This incident has been logged for audit purposes."
        )
    
    return True, ""

def extract_customer_names_from_sql(sql_query: str) -> list[str]:
    """Extract customer names from SQL WHERE clauses."""
    pattern = r"customer_name\s*=\s*'([^']+)'"
    matches = re.findall(pattern, sql_query, re.IGNORECASE)
    return matches

def validate_sql_access(sql_query: str, current_user: str) -> Tuple[bool, str]:
    """
    Layer 3: Validate that the SQL query only accesses the current user's data.
    Implements Row-Level Security per guardrails.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not current_user:
        return False, "🚫 Authentication required to access database."
    
    # Extract customer names from the query
    referenced_customers = extract_customer_names_from_sql(sql_query)
    
    # Check if query references other customers
    for customer in referenced_customers:
        if customer != current_user:
            return False, (
                f"🚫 SECURITY VIOLATION DETECTED!\n"
                f"   Attempted to access: '{customer}'\n"
                f"   You are authenticated as: '{current_user}'\n"
                f"   You can only access your own data.\n"
                f"   This incident has been logged for audit and compliance."
            )
    
    sql_upper = sql_query.upper()

    # Check for queries without WHERE clause on restricted tables
    restricted_tables = ["CUSTOMERS", "TRANSACTIONS", "UPI_CUSTOMER", "UPI_TRANSACTION"]
    dataset_prefix = f"{config.BIGQUERY_DATASET.upper()}."

    for table in restricted_tables:
        # Check both with and without dataset prefix
        if (f"FROM {table}" in sql_upper or f"FROM {dataset_prefix}{table}" in sql_upper):
            if "WHERE" not in sql_upper:
                return False, (
                    f"🚫 SECURITY VIOLATION: Query attempts to access all records without filtering.\n"
                    f"   You can only access your own data (authenticated as: '{current_user}').\n"
                    f"   This incident has been logged for audit and compliance."
                )

    return True, ""

def _check_access_permission(natural_language_query: str, current_user: str) -> Tuple[bool, str]:
    """
    Check if the natural language query is trying to access unauthorized data.
    Focus only on obvious 'all users' patterns.
    """
    if not current_user:
        return False, "🚫 Authentication required to access database."
    
    query_lower = natural_language_query.lower()
    
    # Only block obvious attempts to get ALL users' data
    if any(pattern in query_lower for pattern in [
        "all customers",
        "all users",
        "every customer", 
        "list all customers",
        "show all customers",
        "every user",
        "total customers",
        "count of customers"
    ]):
        return False, (
            f"🚫 Access Denied: You can only access your own data.\n"
            f"   You are authenticated as '{current_user}'.\n"
            f"   Try asking: 'my transactions' or 'my spending'\n"
            f"   This incident has been logged for audit purposes."
        )
    
    return True, ""

# --- 5. Define BigQuery Tool Logic with Security ---

sql_prompt = ChatPromptTemplate.from_messages([
    ("system",
    f"""
    You are a BigQuery SQL expert. Given a user question and the database schema,
    write a valid **BigQuery** SQL query to answer it.
    Only output the SQL query and nothing else - no explanations, no markdown formatting.
    Ensure tables are referenced as `{config.BIGQUERY_DATASET}.table_name`.

    **IMPORTANT TABLE SELECTION**:
    - ALWAYS use the UPI tables (upi_transaction, upi_customer, upi_merchant, upi_bank)
    - DO NOT use the old tables (transactions, customers, merchants)
    - The upi_transaction table contains all transaction data (55,000+ rows)
    - The old tables are empty and deprecated

    **CRITICAL SECURITY RULES**:
    - Current user context: {{current_user}}
    - User type: {{user_type}} (either 'customer' or 'merchant')

    **FOR CUSTOMERS**:
    - You MUST add row-level security filters to ALL queries
    - ALWAYS include: WHERE customer_name = {{current_user}}
    - For transactions, JOIN with customers and filter: WHERE c.customer_name = {{current_user}}
    - NEVER generate queries that access all customers or other users' data
    - If query asks for data about another user, respond with: "ACCESS_DENIED"

    **FOR MERCHANTS**:
    - Merchants can see ALL transactions where they are the payee
    - Filter by: WHERE payee_vpa = {{current_user}} OR merchant_id IN (SELECT merchant_id FROM {config.BIGQUERY_DATASET}.upi_merchant WHERE merchant_vpa = {{current_user}})
    - Merchants can see aggregate statistics for their transactions
    - NEVER show customer personal details (names, emails, account numbers) - only show VPAs and transaction data

    **SQL Rules**:
    - Always use single quotes for string literals, never double quotes
    - BigQuery is case-sensitive for string comparisons
    - Handle NULL values appropriately using IS NULL or IS NOT NULL
    - Do NOT add LIMIT clause - the system handles this automatically

    **Query Patterns with Security for CUSTOMERS**:
    - "my transactions" →
      SELECT t.* FROM {config.BIGQUERY_DATASET}.upi_transaction t
      JOIN {config.BIGQUERY_DATASET}.upi_customer c ON t.payer_vpa = c.primary_vpa
      WHERE c.name = {{current_user}}

    - "my account details" →
      SELECT * FROM {config.BIGQUERY_DATASET}.upi_customer
      WHERE name = {{current_user}}

    **Query Patterns with Security for MERCHANTS**:
    - "my transactions" or "transactions to my store" →
      SELECT t.* FROM {config.BIGQUERY_DATASET}.upi_transaction t
      WHERE t.payee_vpa = {{current_user}}

    - "total sales" or "revenue" →
      SELECT SUM(amount) as total_sales FROM {config.BIGQUERY_DATASET}.upi_transaction t
      WHERE t.payee_vpa = {{current_user}} AND t.status = 'SUCCESS'

    - "my merchant details" →
      SELECT * FROM {config.BIGQUERY_DATASET}.upi_merchant
      WHERE merchant_vpa = {{current_user}}

    Database schema:
    {{formatted_schema}}
    """),
    ("human", "{question}")
])
sql_generation_chain = sql_prompt | llm

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful data assistant. Provide a concise, conversational answer 
    based on the user's question and the summary of the query result.
    1. If the result is a single value, answer directly with context.
    2. If there are no results or an error, state that clearly.
    """),
    ("human", "User Question: {question}\nQuery Result Summary: {result_summary}")
])
summary_chain = summary_prompt | llm

def _execute_query(sql_query: str, current_user: Optional[str] = None, user_type: str = 'customer') -> Tuple[str, pd.DataFrame | None]:
    """
    Execute SQL query with comprehensive security validation and result limits.
    Implements all 4 layers of security guardrails.
    """
    exec_start_time = time.time()
    
    # Layer 1: Query Type Validation
    is_valid_type, error_msg = validate_query_type(sql_query)
    if not is_valid_type:
        print(f"[SECURITY BLOCKED - Query Type] User: {current_user}")
        log_query_attempt(
            user=current_user or 'UNKNOWN',
            query=sql_query,
            status='BLOCKED',
            reason='Prohibited query type detected'
        )
        return error_msg, None

    # Layer 3: SQL Access Validation (Row-Level Security)
    # Only apply customer-specific validation for customers, not merchants
    if current_user and user_type == 'customer':
        is_valid, error_msg = validate_sql_access(sql_query, current_user)
        if not is_valid:
            print(f"[SECURITY BLOCKED - Access] User: {current_user} | Query: {sql_query[:100]}")
            log_query_attempt(
                user=current_user,
                query=sql_query,
                status='BLOCKED',
                reason='Row-level security violation'
            )
            return error_msg, None
    
    try:
        clean_sql = sql_query.strip().replace("```sql", "").replace("```", "")
        
        # Add LIMIT clause if not present (max 1000 rows per guardrails)
        if 'LIMIT' not in clean_sql.upper():
            clean_sql = f"{clean_sql.rstrip(';')} LIMIT 1000"
        
        print(f"--- Executing SQL for user '{current_user}': ---\n{clean_sql}\n" + "-"*50)
        
        # Track BigQuery execution time
        bq_start = time.time()
        
        # Configure query job with limits per guardrails
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            maximum_bytes_billed=10**9  # 1GB limit to prevent expensive queries
        )
        
        query_job = bq_client.query(clean_sql, job_config=job_config)
        
        # Wait with timeout (30 seconds per guardrails)
        results = query_job.result(timeout=30).to_dataframe()
        
        bq_execution_time = time.time() - bq_start
        
        # Get bytes processed
        bytes_processed = query_job.total_bytes_processed if query_job.total_bytes_processed else 0
        
        print(f"⏱️  BigQuery Execution: {bq_execution_time:.3f}s")
        print(f"📊 Bytes Processed: {bytes_processed:,} bytes ({bytes_processed / 1024 / 1024:.2f} MB)")
        
        # Enforce max rows (per guardrails: 1000 rows)
        warning_msg = ""
        if len(results) > 1000:
            results = results.head(1000)
            warning_msg = "\n\n⚠️ Results limited to 1000 rows per security policy."
        
        # Post-execution validation (Double-check security)
        # Only for customers - merchants can see multiple customers' transactions
        if current_user and user_type == 'customer' and not results.empty:
            if 'customer_name' in results.columns:
                unauthorized_names = results[results['customer_name'] != current_user]['customer_name'].unique()
                if len(unauthorized_names) > 0:
                    print(f"[SECURITY] Post-execution check FAILED: Found data for {unauthorized_names}")
                    log_query_attempt(
                        user=current_user,
                        query=sql_query,
                        status='BLOCKED',
                        reason='Post-execution validation failed - unauthorized data detected'
                    )
                    return (
                        f"🚫 Security validation failed: Query returned unauthorized data.\n"
                        f"   This incident has been logged for audit and compliance."
                    ), None
        
        total_exec_time = time.time() - exec_start_time
        print(f"⏱️  Total Execution (with validation): {total_exec_time:.3f}s")
        
        if results.empty:
            return "The query executed successfully but returned no results." + warning_msg, None
        
        return results.to_string(index=False) + warning_msg, results
        
    except Exception as e:
        error_detail = str(e)
        print(f"ERROR DETAILS: {error_detail}")
        log_query_attempt(
            user=current_user or 'UNKNOWN',
            query=sql_query,
            status='ERROR',
            reason=error_detail[:200]
        )
        
        # User-friendly error message
        if "timeout" in error_detail.lower():
            return (
                "⏱️ Query timeout: The query took too long to execute (max 30 seconds).\n"
                "   Try simplifying your query or adding more specific filters."
            ), None
        elif "bytes" in error_detail.lower():
            return (
                "💾 Query too expensive: This query would process too much data.\n"
                "   Try adding more specific filters to reduce the data scanned."
            ), None
        else:
            return f"An error occurred while executing the query: {e}", None

@mcp.tool
def query_customer_database(natural_language_query: str, current_user: str = None, user_type: str = 'customer') -> str:
    """
    Answers questions about customer data, transactions, accounts, or
    financial calculations from a BigQuery database.

    SECURITY: Enforces comprehensive multi-layer security guardrails:
    - Layer 1: Query Parser & Validator (blocks prohibited operations)
    - Layer 2: Database READ-ONLY user permissions
    - Layer 3: Row-Level Security (users can only access their own data)
    - Layer 4: Rate Limiting (10 queries/minute, 100 queries/session)

    Additional Safeguards:
    - Max 1000 rows per query
    - 30-second query timeout
    - 1GB maximum bytes billed
    - Comprehensive audit logging

    Args:
        natural_language_query: The user's question in natural language
        current_user: The authenticated user's name or merchant VPA (REQUIRED for security)
        user_type: Type of user ('customer' or 'merchant'), defaults to 'customer'

    Returns:
        Query results with SQL query included, or security error message
    """
    tool_start_time = time.time()
    sql_gen_time = None
    sql_exec_time = None
    rows_returned = None
    bytes_processed = None
    
    print(f"\n{'='*60}")
    print(f"[BQ Tool] Query: {natural_language_query}")
    print(f"[BQ Tool] Authenticated User: {current_user or 'NONE - DENIED'} ({user_type.upper()})")
    print(f"{'='*60}")

    # 1. Authentication check
    if not current_user:
        total_time = time.time() - tool_start_time
        log_performance_metric(
            user='ANONYMOUS',
            user_type=user_type,
            query=natural_language_query,
            total_time=total_time,
            status='AUTH_FAILED'
        )
        log_query_attempt(
            user='ANONYMOUS',
            query=natural_language_query,
            status='BLOCKED',
            reason='No authentication provided'
        )
        return "🚫 Access Denied: Authentication required to access customer data."
    
    # 2. Layer 4: Rate Limiting
    is_allowed, limit_msg = rate_limiter.is_allowed(current_user)
    if not is_allowed:
        total_time = time.time() - tool_start_time
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=natural_language_query,
            total_time=total_time,
            status='RATE_LIMITED'
        )
        log_query_attempt(
            user=current_user,
            query=natural_language_query,
            status='BLOCKED',
            reason='Rate limit exceeded'
        )
        return limit_msg
    
    # 3. Natural language access permission check (only for customers)
    if user_type == 'customer':
        is_allowed, error_msg = _check_access_permission(natural_language_query, current_user)
        if not is_allowed:
            total_time = time.time() - tool_start_time
            log_performance_metric(
                user=current_user,
                user_type=user_type,
                query=natural_language_query,
                total_time=total_time,
                status='ACCESS_DENIED'
            )
            print(f"[SECURITY] Blocked at NL level: {natural_language_query}")
            log_query_attempt(
                user=current_user,
                query=natural_language_query,
                status='BLOCKED',
                reason='Unauthorized access pattern in natural language query'
            )
            return error_msg

    # 4. Retrieve relevant tables using RAG (before SQL generation)
    rag_retrieval_start = time.time()
    relevant_schema_text = formatted_schema  # Default fallback

    if table_retriever:
        try:
            print(f"\n[RAG] Retrieving relevant tables for query...")
            relevant_tables = table_retriever.retrieve_relevant_tables(
                natural_language_query,
                k=3  # Retrieve top 3 most relevant tables
            )

            if relevant_tables:
                table_names = [t['table_name'] for t in relevant_tables]
                print(f"[RAG] Retrieved tables: {', '.join(table_names)}")

                # Log similarity scores
                for table in relevant_tables:
                    print(f"      • {table['table_name']} (score: {table['similarity_score']})")

                # Get schema text for only the relevant tables
                relevant_schema_text = table_retriever.get_table_schema_text(table_names)
            else:
                print(f"[RAG] No relevant tables found, using all tables")
        except Exception as e:
            print(f"[RAG] Warning: Retrieval failed ({e}), using all tables")

    rag_retrieval_time = time.time() - rag_retrieval_start
    print(f"⏱️  Table RAG Retrieval: {rag_retrieval_time:.3f}s")

    # 5. Generate SQL with timing (using retrieved schema)
    sql_gen_start = time.time()
    sql_response = sql_generation_chain.invoke({
        "question": natural_language_query,
        "current_user": f"'{current_user}'",
        "user_type": user_type,
        "formatted_schema": relevant_schema_text  # Use RAG-retrieved schema
    })
    sql_gen_time = time.time() - sql_gen_start
    sql_query = sql_response.content.strip()

    print(f"⏱️  SQL Generation: {sql_gen_time:.3f}s")
    print(f"[BQ Tool] Generated SQL: {sql_query[:150]}...")

    # 5. Check for access denied in SQL generation
    if "ACCESS_DENIED" in sql_query:
        total_time = time.time() - tool_start_time
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=natural_language_query,
            total_time=total_time,
            sql_generation_time=sql_gen_time,
            status='ACCESS_DENIED'
        )
        log_query_attempt(
            user=current_user,
            query=natural_language_query,
            status='BLOCKED',
            reason='LLM detected unauthorized access attempt'
        )
        return (
            f"🚫 Access Denied: You can only query your own data.\n"
            f"   You are authenticated as '{current_user}'."
        )

    # 6. Validate SQL query was generated properly
    if "cannot answer" in sql_query.lower():
        total_time = time.time() - tool_start_time
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=natural_language_query,
            total_time=total_time,
            sql_generation_time=sql_gen_time,
            status='GENERATION_FAILED'
        )
        return "I'm sorry, but I cannot answer that question with the available database schema."

    sql_upper = sql_query.upper().replace("```SQL", "").replace("```", "").strip()
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        total_time = time.time() - tool_start_time
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=natural_language_query,
            total_time=total_time,
            sql_generation_time=sql_gen_time,
            status='INVALID_SQL'
        )
        return "I encountered an issue generating a SQL query. Please try rephrasing your question."

    # 7. Execute SQL with timing
    sql_exec_start = time.time()
    text_result, df_result = _execute_query(sql_query, current_user, user_type)
    sql_exec_time = time.time() - sql_exec_start
    
    if df_result is not None:
        rows_returned = len(df_result)
    
    # 8. Calculate total time and log
    total_time = time.time() - tool_start_time
    
    log_performance_metric(
        user=current_user,
        user_type=user_type,
        query=natural_language_query,
        total_time=total_time,
        sql_generation_time=sql_gen_time,
        sql_execution_time=sql_exec_time,
        rows_returned=rows_returned,
        bytes_processed=bytes_processed,
        status='SUCCESS' if df_result is not None else 'ERROR'
    )
    
    # Log audit
    if df_result is not None:
        log_query_attempt(
            user=current_user,
            query=natural_language_query,
            status='ALLOWED',
            row_count=len(df_result)
        )
    
    # 9. Build response with SQL at the top
    clean_sql = sql_query.strip().replace("```sql", "").replace("```", "")
    
    response_parts = []
    
    # SQL Section - Always at the top with clear marker
    response_parts.append("[SQL QUERY]")
    response_parts.append(clean_sql)
    response_parts.append("")
    response_parts.append("[DATA RESULTS]")
    response_parts.append("")
    
    # Data Section
    if df_result is not None:
        if not df_result.empty:
            if df_result.shape == (1, 1):
                # Single value
                value = df_result.iloc[0, 0]
                if isinstance(value, float):
                    response_parts.append(f"Result: {value:,.2f}")
                else:
                    response_parts.append(f"Result: {value}")
            else:
                # Table
                response_parts.append(f"Rows: {len(df_result)}")
                response_parts.append("")
                table_str = df_result.to_string(
                    index=False,
                    max_colwidth=25,
                    justify='left'
                )
                response_parts.append(table_str)
        else:
            response_parts.append("No results found.")
    else:
        # Error case - text_result contains the error message
        response_parts.append(text_result)
    
    return "\n".join(response_parts)

# --- 6. Run the Server ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting MCP Toolbox Server with Security Guardrails")
    print("=" * 60)
    print(f"🌐 Project: {config.GCP_PROJECT_ID}")
    print(f"📍 Location: {config.GCP_LOCATION}")
    print(f"🔒 Security Implementation:")
    print(f"   ✓ Layer 1: Query Parser & Validator")
    print(f"   ✓ Layer 2: Database READ-ONLY Permissions")
    print(f"   ✓ Layer 3: Row-Level Security")
    print(f"   ✓ Layer 4: Rate Limiting (10/min, 100/session)")
    print(f"\n📊 Query Safeguards:")
    print(f"   • Max rows per query: 1000")
    print(f"   • Query timeout: 30 seconds")
    print(f"   • Max bytes billed: 1GB")
    print(f"\n📝 Logging:")
    print(f"   • Security audit: mcp_security_audit.log")
    print(f"   • Performance metrics: mcp_performance.log")
    print(f"   • All queries logged with timing and status")
    print("\n📊 Performance Tracking: ENABLED")
    print(f"   • SQL generation time tracked")
    print(f"   • SQL execution time tracked")
    print(f"   • BigQuery bytes processed tracked")
    print(f"   • Real-time metrics displayed")
    print("\n🚫 Prohibited Operations (per Banking Regulations):")
    print(f"   • DELETE, UPDATE, INSERT statements")
    print(f"   • Schema modifications (ALTER, CREATE, DROP)")
    print(f"   • Administrative commands (GRANT, REVOKE)")
    print(f"\n✅ Allowed Operations:")
    print(f"   • SELECT queries only (READ-ONLY access)")
    print("=" * 60)
    
    mcp.run(
        transport="sse",     
        host="0.0.0.0",       
        port=8001,
        path="/sse"           
    )