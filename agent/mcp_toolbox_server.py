import sys
import os
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# No additional imports for performance optimization

from table_context import ACCESS_CONTROL

# Import TableRetriever with proper path handling
try:
    # Try importing from current directory first
    from table_retriever import TableRetriever
except ImportError:
    try:
        # Try importing from agent directory
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from table_retriever import TableRetriever
    except ImportError:
        # If all fails, set TableRetriever to None and continue without it
        print("⚠️  Warning: Could not import TableRetriever. RAG functionality will be disabled.")
        TableRetriever = None

# --- Configure Audit Logging ---
audit_logger = logging.getLogger('mcp_security_audit')
audit_logger.setLevel(logging.INFO)

# Determine absolute path for logs (use project root, not agent subdirectory)
log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audit_log_path = os.path.join(log_dir, 'mcp_security_audit.log')

# File handler for audit trail
audit_handler = logging.FileHandler(audit_log_path)
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

# Determine absolute path for logs (use project root, not agent subdirectory)
perf_log_path = os.path.join(log_dir, 'mcp_performance.log')

perf_handler = logging.FileHandler(perf_log_path)
perf_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
perf_logger.addHandler(perf_handler)

print(f"📝 Performance logs will be written to: {perf_log_path}")

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
    output_formatting_time: float = None,
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
        'query': query[:200],
        'sql_generation_time': round(sql_generation_time, 3) if sql_generation_time else None,
        'sql_execution_time': round(sql_execution_time, 3) if sql_execution_time else None,
        'output_formatting_time': round(output_formatting_time, 3) if output_formatting_time else None,
        'total_time': round(total_time, 3),
        'rows_returned': rows_returned,
        'bytes_processed': bytes_processed,
        'status': status
    }

    # Log only JSON format
    perf_logger.info(json.dumps(metric))

    # Print real-time performance summary to console
    print(f"\n{'─'*60}")
    print(f"⏱️  MCP Tool Performance:")
    if sql_generation_time:
        print(f"   • SQL Generation: {sql_generation_time:.3f}s")
    if sql_execution_time:
        print(f"   • SQL Execution: {sql_execution_time:.3f}s")
    if output_formatting_time:
        print(f"   • Output Formatting: {output_formatting_time:.3f}s")
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
    thinking_budget=0
)

embeddings = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
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
    if TableRetriever is not None:
        table_retriever = TableRetriever()
        rag_initialized = table_retriever.initialize()

        if rag_initialized:
            print("✓ Table RAG retriever initialized successfully")
        else:
            print("⚠️  Table RAG retriever not available - using fallback (all tables)")
            table_retriever = None
    else:
        print("⚠️  TableRetriever not available - using fallback (all tables)")
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

# --- Load Vector Store (Optional - PDFs now in BigQuery) ---
print("[5/5] Loading vector store for PDF queries...")
vector_store = None
try:
    import os
    if os.path.exists(os.path.join(config.VECTOR_STORE_PATH, "index.faiss")):
        vector_store = FAISS.load_local(
            config.VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Local PDF vector store loaded from {config.VECTOR_STORE_PATH}")
    else:
        print(f"ℹ️  Local PDF vector store not found - using BigQuery for all document queries")
        print(f"   All embeddings (PDFs, tables, websites) are now in BigQuery")
except Exception as e:
    print(f"⚠️  Could not load local PDF vector store: {str(e)}")
    print(f"   Using BigQuery for all document queries instead")

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
    Answers questions by searching both PDF documents and website content in BigQuery.
    Use this for questions about:
    - UPI (Unified Payments Interface): how it works, features, security, limits, history
    - Mindgate Solutions: products, services, payment solutions, offerings
    - Digital payment solutions, transaction banking, government solutions
    - Any other topics covered in the knowledge base
    """
    start_time = time.time()

    print(f"[PDF Tool] Received query: {question}")

    try:
        # Track vector search time
        search_start = time.time()

        # Use BigQuery if local vector store is not available
        docs = []
        source_info = []
        
        if vector_store is None:
            # Query BigQuery for PDF embeddings from new table
            try:
                # Import the new vector store class
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from index_new_pdf import NewBigQueryVectorStore
                print(f"📚 New BigQuery vector store imported successfully")
                bq_vector_store = NewBigQueryVectorStore(dataset_name=config.BIGQUERY_DATASET)
                
                # Create query embedding with detailed logging
                print(f"🧠 Creating embedding for query: '{question}'")
                print(f"🔧 Using embedding model: {embeddings.model_name}")
                query_embedding = embeddings.embed_query(question)
                print(f"� Query embedding dimensions: {len(query_embedding)}")
                print(f"📊 Query embedding (first 5 values): {query_embedding[:5]}")
                
                print(f"🔍 Searching new BigQuery table for PDF embeddings with similarity threshold: 0.6")
                results_pdf = bq_vector_store.similarity_search(
                    query_embedding=query_embedding,
                    k=5,
                    similarity_threshold=0.6,
                    source_type="pdf"
                )
                
                print(f"🔍 Searching new BigQuery table for Website embeddings with similarity threshold: 0.6")
                results_web = bq_vector_store.similarity_search(
                    query_embedding=query_embedding,
                    k=5,
                    similarity_threshold=0.6,
                    source_type="website"
                )
                
                # Combine results
                results = results_pdf + results_web
                # Sort by similarity score
                results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                # Take top 5
                results = results[:5]
                
                print(f"📊 BigQuery returned {len(results)} combined results")
                
                # Log detailed results with FULL content
                print(f"\n{'='*80}")
                print(f"📦 RETRIEVED CHUNKS/VECTORS:")
                print(f"{'='*80}")
                for i, result in enumerate(results):
                    print(f"\n📄 CHUNK {i+1}:")
                    print(f"   Source: {result.get('source_name', 'Unknown')}")
                    print(f"   Similarity Score: {result.get('similarity_score', 0):.4f}")
                    print(f"   Content Length: {len(result.get('content', ''))} chars")
                    print(f"\n   FULL CONTENT:")
                    print(f"   {'-'*76}")
                    content = result.get('content', '')
                    # Print content with indentation
                    for line in content.split('\n'):
                        print(f"   {line}")
                    print(f"   {'-'*76}")
                print(f"\n{'='*80}\n")
                
                docs = [type('Doc', (), {'page_content': r['content']}) for r in results]
                # Store source information for later reference
                source_info = [(r.get('source_name', 'Document'), r.get('similarity_score', 0), r['content']) for r in results]
                print(f"📝 Created source_info with {len(source_info)} entries")
            except ImportError as e:
                print(f"⚠️  New BigQuery vector store import failed: {e}")
                import traceback
                traceback.print_exc()
                docs = []
                source_info = []
            except Exception as e:
                print(f"⚠️  New BigQuery vector store search failed: {e}")
                print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                docs = []
                source_info = []
        else:
            # Use local FAISS vector store
            docs = vector_store.similarity_search(question, k=3)
            # For local vector store, create basic source info
            source_info = [(getattr(doc, 'metadata', {}).get('source', 'PDF Document'), 0.0, doc.page_content) for doc in docs]

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
            return "[ANSWER]\nI couldn't find any relevant information in the document to answer that question.\n\n[SOURCES AND EXACT TEXT]\nNo relevant sources found."
        
        print(f"📄 Processing {len(docs)} documents with {len(source_info)} source entries")
        context = "\n\n".join([f"Context {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Track LLM response time
        llm_start = time.time()
        response = pdf_generation_chain.invoke({
            "context": context,
            "question": question
        })
        llm_time = time.time() - llm_start

        # Build response - only include the answer, not the sources
        formatting_start = time.time()

        # Return only the answer without the sources section
        final_response = response.content

        output_format_time = time.time() - formatting_start
        total_time = time.time() - start_time

        log_performance_metric(
            user='SYSTEM',
            user_type='pdf_query',
            query=question,
            total_time=total_time,
            pdf_search_time=search_time + llm_time,
            output_formatting_time=output_format_time,
            rows_returned=len(docs),
            status='SUCCESS'
        )

        print(f"⏱️  PDF Query completed in {total_time:.3f}s (Search: {search_time:.3f}s, LLM: {llm_time:.3f}s, Formatting: {output_format_time:.3f}s)")

        return final_response
        
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
            "   This incident has been logged for audit and compliance."
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

    **DATE FILTERING RULES**:
    - Only add date filters when the user explicitly requests a time range
    - For general queries like "show my transactions", do NOT add date filters
    - Always sort by most recent: `ORDER BY initiated_at DESC`
    - Examples:
      * User: "show my transactions" → No date filter, just ORDER BY
      * User: "recent transactions" → Add LIMIT 100, ORDER BY initiated_at DESC
      * User: "last 30 days" → Add `WHERE initiated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)`
      * User: "transactions in January 2025" → Use specific date range

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
    - "my transactions" or "show my transactions" →
      SELECT t.* FROM {config.BIGQUERY_DATASET}.upi_transaction t
      JOIN {config.BIGQUERY_DATASET}.upi_customer c ON t.payer_vpa = c.primary_vpa
      WHERE c.name = {{current_user}}
      ORDER BY t.initiated_at DESC

    - "my recent transactions" →
      SELECT t.* FROM {config.BIGQUERY_DATASET}.upi_transaction t
      JOIN {config.BIGQUERY_DATASET}.upi_customer c ON t.payer_vpa = c.primary_vpa
      WHERE c.name = {{current_user}}
      ORDER BY t.initiated_at DESC
      LIMIT 100

    - "my account details" →
      SELECT * FROM {config.BIGQUERY_DATASET}.upi_customer
      WHERE name = {{current_user}}

    **Query Patterns with Security for MERCHANTS**:
    - "my transactions" or "transactions to my store" →
      SELECT t.* FROM {config.BIGQUERY_DATASET}.upi_transaction t
      WHERE t.payee_vpa = {{current_user}}
      ORDER BY t.initiated_at DESC

    - "my recent transactions" →
      SELECT t.* FROM {config.BIGQUERY_DATASET}.upi_transaction t
      WHERE t.payee_vpa = {{current_user}}
      ORDER BY t.initiated_at DESC
      LIMIT 100

    - "total sales" or "revenue" (all time) →
      SELECT SUM(amount) as total_sales FROM {config.BIGQUERY_DATASET}.upi_transaction t
      WHERE t.payee_vpa = {{current_user}} AND t.status = 'SUCCESS'

    - "sales in last 30 days" →
      SELECT SUM(amount) as total_sales FROM {config.BIGQUERY_DATASET}.upi_transaction t
      WHERE t.payee_vpa = {{current_user}} AND t.status = 'SUCCESS'
      AND t.initiated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)

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

    # 4. Use full schema (RAG removed)
    relevant_schema_text = formatted_schema
    print(f"[RAG] Using full schema ({len(schema_info)} tables)")

    # 5. Generate SQL with LLM
    sql_gen_start = time.time()
    
    # Generate SQL with LLM (using retrieved schema)
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

    # Clean SQL query
    sql_query = sql_query.replace("```SQL", "").replace("```", "").strip()
    
    sql_upper = sql_query.upper()
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH') or sql_upper.startswith('/*')):
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

    # 8. Return ONLY metadata - let the agent format the data
    # This prevents duplicate output
    formatting_start = time.time()

    clean_sql = sql_query.strip().replace("```sql", "").replace("```", "")

    # Return minimal metadata - agent will handle formatting
    if df_result is not None and not df_result.empty:
        # Convert to simple list of dicts
        data = df_result.to_dict('records')

        # Return structured data for agent to format
        # Use a format that tells the agent what to do WITHOUT showing raw data
        response = f"Query executed successfully. SQL: {clean_sql}\n\nFound {len(data)} transactions. Present them in a clear numbered list format with the SQL query in a code block."
    elif df_result is not None:
        response = f"Query executed successfully. SQL: {clean_sql}\n\nNo results found."
    else:
        # Error case
        response = text_result

    output_format_time = time.time() - formatting_start

    # 9. Calculate total time and log
    total_time = time.time() - tool_start_time

    log_performance_metric(
        user=current_user,
        user_type=user_type,
        query=natural_language_query,
        total_time=total_time,
        sql_generation_time=sql_gen_time,
        sql_execution_time=sql_exec_time,
        output_formatting_time=output_format_time,
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

    return response

@mcp.tool
async def generate_sql_for_query(
    natural_language_query: str,
    current_user: str = None,
    user_type: str = 'customer'
) -> str:
    """
    Step 1: Generate SQL query from natural language.
    Returns SQL wrapped in markdown code block, ready to display.
    """
    tool_start_time = time.time()
    sql_gen_time = None

    print(f"\n{'='*60}")
    print(f"[SQL Gen] Query: {natural_language_query}")
    print(f"[SQL Gen] User: {current_user or 'NONE'} ({user_type.upper()})")
    print(f"{'='*60}")

    # 1. Authentication check
    if not current_user:
        return "🚫 Access Denied: Authentication required to access customer data."
    
    # 2. Rate Limiting
    is_allowed, limit_msg = rate_limiter.is_allowed(current_user)
    if not is_allowed:
        log_query_attempt(current_user, natural_language_query, 'BLOCKED', 'Rate limit exceeded')
        return limit_msg
    
    # 3. Access permission check (customers only)
    if user_type == 'customer':
        is_allowed, error_msg = _check_access_permission(natural_language_query, current_user)
        if not is_allowed:
            log_query_attempt(current_user, natural_language_query, 'BLOCKED', 'Unauthorized access pattern')
            return error_msg

    # 4. Use full schema
    relevant_schema_text = formatted_schema
    print(f"[RAG] Using full schema ({len(schema_info)} tables)")

    # 5. Generate SQL
    sql_gen_start = time.time()
    
    result = await sql_generation_chain.ainvoke({
        "question": natural_language_query,
        "current_user": f"'{current_user}'",
        "user_type": user_type,
        "formatted_schema": relevant_schema_text
    })
    
    sql_query = result.content.strip() if hasattr(result, 'content') else str(result).strip()
    sql_gen_time = time.time() - sql_gen_start
    print(f"⏱️  SQL Generation: {sql_gen_time:.3f}s")

    # Check for access denied
    if "ACCESS_DENIED" in sql_query:
        log_query_attempt(current_user, natural_language_query, 'BLOCKED', 'LLM detected unauthorized access')
        log_performance_metric(current_user, user_type, natural_language_query, time.time() - tool_start_time, sql_gen_time, status='ACCESS_DENIED')
        return "🚫 Access Denied: You can only query your own data."

    # Validate SQL
    if "cannot answer" in sql_query.lower():
        log_performance_metric(current_user, user_type, natural_language_query, time.time() - tool_start_time, sql_gen_time, status='GENERATION_FAILED')
        return "I'm sorry, but I cannot answer that question with the available database schema."

    # Clean SQL query - remove any existing code block markers
    sql_query = sql_query.replace("```SQL", "").replace("```sql", "").replace("```", "").strip()
    sql_upper = sql_query.upper()
    
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH') or sql_upper.startswith('/*')):
        log_performance_metric(current_user, user_type, natural_language_query, time.time() - tool_start_time, sql_gen_time, status='INVALID_SQL')
        return "I encountered an issue generating a SQL query. Please try rephrasing your question."

    # Security validation
    is_valid_type, error_msg = validate_query_type(sql_query)
    if not is_valid_type:
        log_query_attempt(current_user, sql_query, 'BLOCKED', 'Prohibited query type detected')
        log_performance_metric(current_user, user_type, natural_language_query, time.time() - tool_start_time, sql_gen_time, status='BLOCKED_TYPE')
        return error_msg

    if current_user and user_type == 'customer':
        is_valid, error_msg = validate_sql_access(sql_query, current_user)
        if not is_valid:
            log_query_attempt(current_user, sql_query, 'BLOCKED', 'Row-level security violation')
            log_performance_metric(current_user, user_type, natural_language_query, time.time() - tool_start_time, sql_gen_time, status='BLOCKED_ACCESS')
            return error_msg

    total_time = time.time() - tool_start_time

    # Log performance
    log_performance_metric(
        user=current_user,
        user_type=user_type,
        query=natural_language_query,
        total_time=total_time,
        sql_generation_time=sql_gen_time,
        status='SUCCESS'
    )
    
    # Return SQL in code block - agent should display as-is without adding another code block
    return sql_query


@mcp.tool
async def execute_sql_query(
    sql_query: str,
    current_user: str = None,
    user_type: str = 'customer'
) -> str:
    """
    Step 2: Execute SQL and return formatted results.
    Returns only the formatted transaction list as a string (no header/footer).
    """
    exec_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[SQL Exec] Executing for: {current_user} ({user_type.upper()})")
    print(f"{'='*60}")

    if not current_user:
        return "🚫 Authentication required"

    try:
        # Add LIMIT if not present
        clean_sql = sql_query.strip()
        if 'LIMIT' not in clean_sql.upper():
            clean_sql = f"{clean_sql.rstrip(';')} LIMIT 1000"
        
        print(f"--- Executing SQL ---\n{clean_sql}\n" + "-"*50)
        
        # Configure query job
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            maximum_bytes_billed=10**9
        )
        
        bq_start = time.time()
        query_job = bq_client.query(clean_sql, job_config=job_config)
        
        # Fetch results
        results = []
        column_names = None
        row_count = 0
        
        for row in query_job.result(page_size=100):
            if column_names is None:
                column_names = list(row.keys())
            
            row_data = {col: str(row[col]) for col in column_names}
            results.append(row_data)
            row_count += 1
        
        bq_time = time.time() - bq_start

        print(f"⏱️  BigQuery Execution: {bq_time:.3f}s")

        if row_count == 0:
            exec_time = time.time() - exec_start_time

            log_performance_metric(
                user=current_user,
                user_type=user_type,
                query=sql_query[:200],
                total_time=exec_time,
                sql_execution_time=bq_time,
                rows_returned=0,
                status='SUCCESS'
            )

            return "No results found."

        # Format results - clean output without header/footer
        formatting_start = time.time()

        result_lines = []

        # Format each row
        for idx, row in enumerate(results, 1):
            result_lines.append(f"\n🔹 Transaction #{idx}\n")
            result_lines.append("-" * 80 + "\n")

            for col, value in row.items():
                if 'amount' in col.lower():
                    result_lines.append(f"  💰 {col}: ₹{value}\n")
                elif 'date' in col.lower() or 'at' in col.lower():
                    result_lines.append(f"  📅 {col}: {value}\n")
                elif 'status' in col.lower():
                    emoji = "✅" if value == "SUCCESS" else "❌" if value == "FAILED" else "⏳"
                    result_lines.append(f"  {emoji} {col}: {value}\n")
                elif 'id' in col.lower():
                    result_lines.append(f"  🆔 {col}: {value}\n")
                else:
                    result_lines.append(f"  • {col}: {value}\n")

        output_format_time = time.time() - formatting_start
        exec_time = time.time() - exec_start_time

        log_query_attempt(current_user, sql_query, 'ALLOWED', row_count=row_count)

        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=sql_query[:200],
            total_time=exec_time,
            sql_execution_time=bq_time,
            output_formatting_time=output_format_time,
            rows_returned=row_count,
            bytes_processed=query_job.total_bytes_processed if query_job.total_bytes_processed else 0,
            status='SUCCESS'
        )
        
        # Return ONLY the formatted transactions - no header/footer
        return "".join(result_lines)
        
    except Exception as e:
        exec_time = time.time() - exec_start_time
        error_msg = f"❌ Error executing query: {str(e)}"
        
        log_query_attempt(current_user, sql_query, 'ERROR', reason=str(e))
        
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=sql_query[:200],
            total_time=exec_time,
            rows_returned=0,
            status='ERROR'
        )
        
        return error_msg

@mcp.tool
async def execute_sql_query_streaming(
    sql_query: str,
    current_user: str = None,
    user_type: str = 'customer'
):
    """
    Step 2: Execute SQL and stream results line-by-line for real-time display.
    This version yields results as they're fetched from BigQuery.
    """
    exec_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[SQL Exec Streaming] Executing for: {current_user} ({user_type.upper()})")
    print(f"{'='*60}")

    if not current_user:
        yield {
            "status": "error",
            "message": "🚫 Authentication required"
        }
        return  # No value here - just stop the generator

    try:
        # Add LIMIT if not present
        clean_sql = sql_query.strip()
        if 'LIMIT' not in clean_sql.upper():
            clean_sql = f"{clean_sql.rstrip(';')} LIMIT 1000"
        
        print(f"--- Executing SQL (Streaming) ---\n{clean_sql}\n" + "-"*50)
        
        # Configure query job
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            maximum_bytes_billed=10**9
        )
        
        bq_start = time.time()
        query_job = bq_client.query(clean_sql, job_config=job_config)
        
        # Yield header immediately
        yield {
            "type": "header",
            "message": "📊 Query executing...\n",
            "sql_query": clean_sql
        }
        
        # Stream results as they arrive
        row_count = 0
        column_names = None
        
        # Yield intro
        yield {
            "type": "intro",
            "message": "=" * 100 + "\n"
        }
        
        for row in query_job.result(page_size=10):  # Fetch in small batches
            if column_names is None:
                column_names = list(row.keys())
                # Yield column headers
                yield {
                    "type": "columns",
                    "columns": column_names,
                    "message": f"📋 Columns: {', '.join(column_names)}\n\n"
                }
            
            row_count += 1
            
            # Format row data
            row_data = {}
            row_text = f"\n🔹 Transaction #{row_count}\n"
            row_text += "-" * 100 + "\n"
            
            for col in column_names:
                value = str(row[col])
                row_data[col] = value
                
                if 'amount' in col.lower():
                    row_text += f"  💰 {col}: ₹{value}\n"
                elif 'date' in col.lower() or 'at' in col.lower():
                    row_text += f"  📅 {col}: {value}\n"
                elif 'status' in col.lower():
                    emoji = "✅" if value == "SUCCESS" else "❌" if value == "FAILED" else "⏳"
                    row_text += f"  {emoji} {col}: {value}\n"
                elif 'id' in col.lower():
                    row_text += f"  🆔 {col}: {value}\n"
                else:
                    row_text += f"  • {col}: {value}\n"
            
            # Yield this row immediately
            yield {
                "type": "row",
                "row_number": row_count,
                "data": row_data,
                "message": row_text
            }
        
        bq_time = time.time() - bq_start
        exec_time = time.time() - exec_start_time
        
        # Yield summary
        summary_text = "\n" + "=" * 100 + "\n"
        summary_text += f"\n✅ Total: {row_count} transaction(s)\n"
        summary_text += f"⏱️  Execution time: {exec_time:.3f}s\n"
        summary_text += f"📊 Data processed: {query_job.total_bytes_processed or 0:,} bytes\n"
        
        yield {
            "type": "summary",
            "row_count": row_count,
            "execution_time": exec_time,
            "bq_execution_time": bq_time,
            "bytes_processed": query_job.total_bytes_processed if query_job.total_bytes_processed else 0,
            "message": summary_text
        }
        
        # Log audit
        log_query_attempt(current_user, sql_query, 'ALLOWED', row_count=row_count)
        
        # Log performance
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=sql_query[:200],
            total_time=exec_time,
            sql_execution_time=bq_time,
            rows_returned=row_count,
            bytes_processed=query_job.total_bytes_processed if query_job.total_bytes_processed else 0,
            status='SUCCESS'
        )
        
    except Exception as e:
        exec_time = time.time() - exec_start_time
        error_msg = f"❌ Error executing query: {str(e)}"
        
        log_query_attempt(current_user, sql_query, 'ERROR', error=str(e))
        
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=sql_query[:200],
            total_time=exec_time,
            rows_returned=0,
            status='ERROR'
        )
        
        yield {
            "type": "error",
            "execution_time": exec_time,
            "message": error_msg
        }
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
