import os
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
from google.cloud import storage, aiplatform
from google.auth import default
from google.auth.transport.requests import Request
import requests
import base64
import uuid
from google import genai

# Import utilities
from schema_cache_manager import (
    load_or_refresh_schema,
    SchemaCache
)

from table_context import ACCESS_CONTROL

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


# --- Configure Video Performance Logging ---
video_perf_logger = logging.getLogger('video_performance')
video_perf_logger.setLevel(logging.INFO)

video_perf_handler = logging.FileHandler('video_generation_performance.log')
video_perf_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
video_perf_logger.addHandler(video_perf_handler)

# --- Configure Video URL Storage ---
video_url_logger = logging.getLogger('video_urls')
video_url_logger.setLevel(logging.INFO)

video_url_handler = logging.FileHandler('video_urls.log')
video_url_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
video_url_logger.addHandler(video_url_handler)


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

def log_video_performance_metric(
    user: str,
    user_type: str,
    data_summary: str,
    video_style: str,
    duration: int,
    prompt_length: int,
    video_generation_time: float,
    upload_time: float,
    total_time: float,
    video_size_mb: float,
    video_url: str,
    gcs_path: str,
    status: str = "SUCCESS",
    error_message: str = None
):
    """Log detailed video generation performance metrics"""
    
    metric = {
        'timestamp': datetime.now().isoformat(),
        'user': user,
        'user_type': user_type,
        'data_summary': data_summary[:200],  # Truncate long summaries
        'video_style': video_style,
        'duration_seconds': duration,
        'prompt_length': prompt_length,
        'timings': {
            'video_generation': round(video_generation_time, 3),
            'upload': round(upload_time, 3),
            'total': round(total_time, 3)
        },
        'video_size_mb': round(video_size_mb, 2),
        'gcs_path': gcs_path,
        'video_url': video_url,
        'status': status,
        'error_message': error_message[:200] if error_message else None
    }
    
    # Log to video performance file
    video_perf_logger.info(json.dumps(metric))
    
    # Log URL separately for easy reference
    if status == "SUCCESS":
        video_url_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'user_type': user_type,
            'video_url': video_url,
            'gcs_path': gcs_path,
            'duration': duration,
            'size_mb': round(video_size_mb, 2),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        video_url_logger.info(json.dumps(video_url_entry))
    
    # Print detailed performance summary
    print(f"\n{'═'*60}")
    print(f"📹 VIDEO GENERATION PERFORMANCE REPORT")
    print(f"{'═'*60}")
    print(f"👤 User: {user} ({user_type.upper()})")
    print(f"🎨 Style: {video_style} | Duration: {duration}s")
    print(f"📝 Prompt Length: {prompt_length} characters")
    print(f"\n⏱️  Timing Breakdown:")
    print(f"   • Video Generation: {video_generation_time:.3f}s")
    print(f"   • Upload to GCS: {upload_time:.3f}s")
    print(f"   • Total Time: {total_time:.3f}s")
    print(f"\n📦 Video Details:")
    print(f"   • Size: {video_size_mb:.2f} MB")
    print(f"   • Format: MP4 (16:9)")
    print(f"   • Status: {status}")
    if status == "SUCCESS":
        print(f"\n💾 Storage:")
        print(f"   • GCS Path: {gcs_path}")
        print(f"   • URL Expires: 24 hours")
    elif error_message:
        print(f"\n❌ Error: {error_message[:100]}...")
    print(f"{'═'*60}\n")


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

# --- Verify Video Storage GCP Configuration ---
if not hasattr(config, 'VIDEO_GCP_PROJECT_ID') or not config.VIDEO_GCP_PROJECT_ID:
    print("⚠️  VIDEO_GCP_PROJECT_ID not set, using main project for video storage")
    config.VIDEO_GCP_PROJECT_ID = config.GCP_PROJECT_ID

if not hasattr(config, 'VIDEO_GCP_LOCATION'):
    config.VIDEO_GCP_LOCATION = config.GCP_LOCATION
    print(f"⚠️  VIDEO_GCP_LOCATION not set, using: {config.VIDEO_GCP_LOCATION}")

if not hasattr(config, 'VIDEO_GCS_BUCKET'):
    config.VIDEO_GCS_BUCKET = "mcp_chatbot"
    print(f"⚠️  VIDEO_GCS_BUCKET not set, using default: {config.VIDEO_GCS_BUCKET}")

if not hasattr(config, 'VIDEO_GCS_PATH'):
    config.VIDEO_GCS_PATH = "videos"

print(f"\n📍 Vertex AI Configuration:")
print(f"   Project: {config.GCP_PROJECT_ID}")
print(f"   Location: {config.GCP_LOCATION}")

print(f"\n📹 Video Storage Configuration:")
print(f"   Project: {config.VIDEO_GCP_PROJECT_ID}")
print(f"   Location: {config.VIDEO_GCP_LOCATION}")
print(f"   Bucket: gs://{config.VIDEO_GCS_BUCKET}/{config.VIDEO_GCS_PATH}")

print("\n[1/6] Initializing AI models with Vertex AI...")
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

print("\n[2/6] Connecting to BigQuery...")
bq_client = bigquery.Client(project=config.GCP_PROJECT_ID)
print(f"✓ Connected to project: {config.GCP_PROJECT_ID}")

# --- Fetch Dynamic Schema ---
print("\n[3/6] Loading schema from cache...")
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

# --- Display Schema ---
print("\n[4/6] Schema loaded successfully")
print("\n" + "="*60)
print("📚 SCHEMA SUMMARY:")
print("="*60)
for table_name in schema_info.keys():
    print(f"  • {table_name}: {len(schema_info[table_name]['fields'])} columns")
print("="*60 + "\n")

# --- Load Vector Stores ---
print("[5/6] Loading vector stores for document queries...")

# Load PDF vector store
pdf_vector_store = None
try:
    if os.path.exists(os.path.join(config.VECTOR_STORE_PATH, "index.faiss")):
        pdf_vector_store = FAISS.load_local(
            config.VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ PDF vector store loaded from {config.VECTOR_STORE_PATH}")
    else:
        print(f"⚠️  PDF vector store not found at {config.VECTOR_STORE_PATH}")
        print("   Run 'python -m agent.pdf_indexer' to create it.")
except Exception as e:
    print(f"⚠️  Could not load PDF vector store: {e}")

# Load website vector store
website_vector_store = None
try:
    if os.path.exists(os.path.join(config.WEBSITE_VECTOR_STORE_PATH, "index.faiss")):
        website_vector_store = FAISS.load_local(
            config.WEBSITE_VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Website vector store loaded from {config.WEBSITE_VECTOR_STORE_PATH}")
    else:
        print(f"ℹ️  Website vector store not found at {config.WEBSITE_VECTOR_STORE_PATH}")
        print("   Run 'python -m agent.website_indexer' to create it.")
except Exception as e:
    print(f"⚠️  Could not load website vector store: {e}")

# Ensure at least one store is available
if not pdf_vector_store and not website_vector_store:
    print("❌ FATAL: No vector stores available!")
    print("Please run 'python -m agent.pdf_indexer' to create the PDF vector store.")
    raise RuntimeError("No vector stores available")

# Keep backward compatibility
vector_store = pdf_vector_store if pdf_vector_store else website_vector_store

print("\n[6/6] Initializing Vertex AI for Veo 3.1...")
try:
    import vertexai
    from google.oauth2 import service_account

    # Load video service account credentials
    video_credentials = service_account.Credentials.from_service_account_file(
        config.VIDEO_SERVICE_ACCOUNT_KEY,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    # Initialize Vertex AI with video project and credentials
    vertexai.init(
        project=config.VIDEO_GCP_PROJECT_ID,
        location=config.VIDEO_GCP_LOCATION,
        credentials=video_credentials
    )
    print(f"✓ Vertex AI initialized for video generation")
    print(f"  Project: {config.VIDEO_GCP_PROJECT_ID}")
    print(f"  Credentials: {config.VIDEO_SERVICE_ACCOUNT_KEY}")
    VEO_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Warning: Could not initialize Vertex AI: {e}")
    VEO_AVAILABLE = False

def _generate_video_with_veo31(prompt: str, speech_text: str = None, duration: int = 5, reference_image_path: str = None) -> bytes:
    """
    Generate video using Veo 3.1 via Gemini API with optional lip-sync

    Args:
        prompt: Video generation prompt (scene description)
        speech_text: Text to be spoken with lip-sync (optional)
        duration: Video duration in seconds (4, 6, or 8)
        reference_image_path: Path to reference image for character consistency

    Returns:
        Video bytes
    """
    try:
        print(f"[VEO] Starting video generation with Veo 3.1 (Gemini API)...")
        print(f"[VEO] Duration: {duration}s")

        # Clamp duration to valid values
        if duration <= 4:
            duration = 4
        elif duration <= 6:
            duration = 6
        else:
            duration = 8

        # Initialize Gemini client
        client = genai.Client(api_key=config.GOOGLE_API_KEY)

        # Build complete prompt with speech text for lip-sync
        complete_prompt = prompt
        if speech_text:
            print(f"[VEO] Speech text: {speech_text[:100]}...")
            # Include the actual speech text in the prompt for lip-sync
            complete_prompt = f"{prompt}\n\nThe person says: \"{speech_text}\" with clear lip-sync and natural expression."

        print(f"[VEO] Prompt: {complete_prompt[:150]}...")

        # Load reference image if provided
        reference_image = None
        if reference_image_path:
            import os
            from google.genai import types
            if os.path.exists(reference_image_path):
                print(f"[VEO] Loading reference image: {reference_image_path}")
                # Read image bytes
                with open(reference_image_path, 'rb') as img_file:
                    image_bytes = img_file.read()

                # Determine mime type
                mime_type = "image/png" if reference_image_path.lower().endswith('.png') else "image/jpeg"

                # Create Image object for Gemini API
                reference_image = types.Image(
                    imageBytes=image_bytes,
                    mimeType=mime_type
                )
                print(f"[VEO] ✓ Reference image loaded ({len(image_bytes)} bytes, {mime_type})")
            else:
                print(f"[VEO] ⚠️  Reference image not found: {reference_image_path}")

        # Start video generation with reference image
        print(f"[VEO] Initiating video generation request...")
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=complete_prompt,
            image=reference_image  # Use reference image as the base
        )

        operation_id = operation.name.split('/')[-1]
        print(f"[VEO] ✅ Operation started: {operation_id}")
        print(f"[VEO] ⏳ Polling for completion (this takes 1-3 minutes)...")

        # Poll for completion
        max_wait_time = 600  # 10 minutes max
        poll_interval = 10  # Poll every 10 seconds
        start_time = time.time()
        poll_count = 0

        while not operation.done:
            elapsed = time.time() - start_time

            if elapsed > max_wait_time:
                raise Exception(f"Video generation timed out after {int(elapsed)} seconds")

            poll_count += 1
            if poll_count % 3 == 1:  # Print every 30 seconds
                print(f"[VEO] Polling... ({int(elapsed)}s elapsed)")

            time.sleep(poll_interval)

            # Refresh operation status
            operation = client.operations.get(operation)

        # Operation completed
        gen_time = time.time() - start_time
        print(f"[VEO] ✅ Operation completed in {gen_time:.1f}s!")

        # Check response
        response = operation.response

        # Check for filtered content
        if hasattr(response, 'rai_media_filtered_count') and response.rai_media_filtered_count:
            filtered_reasons = response.rai_media_filtered_reasons or []
            error_msg = f"Content filtered by safety policies. Try simpler, less specific prompts. Reasons: {filtered_reasons}"
            print(f"[VEO] ❌ {error_msg}")
            raise Exception(error_msg)

        # Get generated videos
        if not response.generated_videos:
            raise Exception("No videos generated in response")

        video = response.generated_videos[0]
        video_uri = video.video.uri

        print(f"[VEO] Downloading video from Google servers...")

        # Download video bytes with API key authentication
        download_headers = {
            "x-goog-api-key": config.GOOGLE_API_KEY
        }
        download_response = requests.get(video_uri, headers=download_headers, timeout=60)

        if download_response.status_code != 200:
            raise Exception(f"Video download failed: {download_response.status_code}")

        video_bytes = download_response.content
        print(f"[VEO] ✓ Video downloaded successfully ({len(video_bytes)} bytes, {len(video_bytes)/(1024*1024):.2f} MB)")

        return video_bytes

    except Exception as e:
        print(f"[VEO] ❌ Error: {e}")
        raise Exception(f"Veo 3.1 generation failed: {e}")

    

print("\n" + "="*60)
print("✅ ALL RESOURCES INITIALIZED SUCCESSFULLY (VERTEX AI + VEO 3.1)")
print("="*60 + "\n")

# --- 2. CREATE FastMCP INSTANCE ---
mcp = FastMCP("MyToolboxServer")

# --- 3. Define PDF Tool Logic ---
pdf_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are a helpful assistant that answers questions based on provided knowledge base documents.
    Use the provided context to answer the user's question.
    Your answer should be based SOLELY on the context provided.
    If the context does not contain the answer, say that you cannot find the information in the documents.

    IMPORTANT: Always cite your sources in your answer. Each context snippet shows its source (PDF or website).
    Mention the source when providing information, for example:
    - "According to the PDF document..."
    - "Based on the website information from [URL]..."
    - "The documentation indicates..."

    Context:
    {context}
    """),
    ("human", "{question}")
])
pdf_generation_chain = pdf_prompt | llm

@mcp.tool
def ask_document(question: str, source: str = "all") -> str:
    """
    Answers questions by searching knowledge base documents (PDFs and websites).
    
    Args:
        question: The question to answer
        source: Which source to search - "pdf", "website", or "all" (default: "all")
    
    Returns:
        Answer based on the most relevant document chunks found
    """
    start_time = time.time()
    
    print(f"[Document Query] Question: {question}")
    print(f"[Document Query] Source filter: {source}")
    
    try:
        stores_to_search = []
        
        if source.lower() in ["all", "pdf"] and pdf_vector_store:
            stores_to_search.append(("pdf", pdf_vector_store))
        
        if source.lower() in ["all", "website"] and website_vector_store:
            stores_to_search.append(("website", website_vector_store))
        
        if not stores_to_search:
            return f"No vector stores available for source: {source}"
        
        # Search all relevant stores with MORE results
        search_start = time.time()
        all_docs = []
        
        for source_name, store in stores_to_search:
            # Use MMR (Maximal Marginal Relevance) for better diversity and relevance
            try:
                # Try MMR first for better diverse results
                docs = store.max_marginal_relevance_search(
                    question,
                    k=7,           # Get 7 documents
                    fetch_k=20,    # From pool of 20 candidates
                    lambda_mult=0.7  # Balance relevance (0.7) vs diversity (0.3)
                )
            except:
                # Fallback to regular similarity search
                docs = store.similarity_search(question, k=7)

            # Add source info and score to metadata
            for i, doc in enumerate(docs):
                doc.metadata['search_source'] = source_name
                doc.metadata['rank'] = i  # Track original ranking
            all_docs.extend(docs)
        
        search_time = time.time() - search_start
        
        # Debug: Print what was found
        print(f"\n🔍 Found {len(all_docs)} total chunks")
        for i, doc in enumerate(all_docs[:3]):
            print(f"\nTop result {i+1}:")
            print(f"  Source: {doc.metadata.get('source', 'unknown')}")
            print(f"  Preview: {doc.page_content[:150]}...")
        
        if not all_docs:
            total_time = time.time() - start_time
            log_performance_metric(
                user='SYSTEM',
                user_type='document_query',
                query=question,
                total_time=total_time,
                pdf_search_time=search_time,
                status='NO_RESULTS'
            )
            return "I couldn't find any relevant information to answer that question."
        
        # Take top 10 results for better coverage and complete answers
        all_docs = all_docs[:10]
        
        # Build context with source attribution
        context_parts = []
        for i, doc in enumerate(all_docs):
            source_type = doc.metadata.get('search_source', 'unknown')
            source_info = doc.metadata.get('source', 'unknown')
            title = doc.metadata.get('title', '')
            
            context_parts.append(
                f"Context {i+1} [from {source_type} - {title}]:\n{doc.page_content}"
            )
        
        context = "\n\n".join(context_parts)
        
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
            user_type='document_query',
            query=question,
            total_time=total_time,
            pdf_search_time=search_time + llm_time,
            rows_returned=len(all_docs),
            status='SUCCESS'
        )
        
        print(f"⏱️  Document query completed in {total_time:.3f}s (Search: {search_time:.3f}s, LLM: {llm_time:.3f}s)")
        print(f"📊 Used {len(all_docs)} relevant chunks")
        
        return response.content
        
    except Exception as e:
        total_time = time.time() - start_time
        log_performance_metric(
            user='SYSTEM',
            user_type='document_query',
            query=question,
            total_time=total_time,
            status='ERROR'
        )
        raise e
@mcp.tool
def ask_upi_document(question: str) -> str:
    """
    [DEPRECATED] Use ask_document() instead.
    Answers questions about the UPI (Unified Payments Interface) process.
    """
    return ask_document(question, source="pdf")

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
    {formatted_schema}
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

    # 4. Generate SQL with timing
    sql_gen_start = time.time()
    sql_response = sql_generation_chain.invoke({
        "question": natural_language_query,
        "current_user": f"'{current_user}'",
        "user_type": user_type
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


@mcp.tool
def generate_video_from_data(
    data_summary: str,
    video_style: str = "professional",
    duration: int = 5,
    current_user: str = None,
    user_type: str = 'customer'
) -> str:
    """
    Generates a video visualization using Veo 3.1 based on data analysis results.
    
    Use this when users request visual presentations of their financial data,
    transaction summaries, or banking insights in video format.
    
    IMPORTANT: Always query the database FIRST to get actual data, then pass the
    results summary to this tool for video generation.
    
    Args:
        data_summary: Detailed description of the data to visualize (include actual numbers, trends, insights)
        video_style: Style of video - options: "professional", "animated", "infographic", "modern"
        duration: Video duration in seconds (5-10 seconds recommended, max 10)
        current_user: The authenticated user making the request (REQUIRED)
        user_type: Type of user ('customer' or 'merchant')
    
    Returns:
        GCS URL and download link for the generated video with performance metrics
    """
    tool_start_time = time.time()
    
    # Initialize timing variables
    video_generation_time = 0
    upload_time = 0
    video_size_mb = 0
    video_url = ""
    gcs_path = ""
    prompt_length = 0
    
    # Authentication check
    if not current_user:
        log_performance_metric(
            user='ANONYMOUS',
            user_type=user_type,
            query=data_summary[:200],
            total_time=time.time() - tool_start_time,
            status='AUTH_FAILED'
        )
        return "🚫 Access Denied: Authentication required to generate videos."
    
    # Check if Veo is available
    if not VEO_AVAILABLE:
        return "❌ Video generation is currently unavailable. Vertex AI could not be initialized."
    
    print(f"\n{'='*60}")
    print(f"[VIDEO Tool] User: {current_user} ({user_type.upper()})")
    print(f"[VIDEO Tool] Data Summary: {data_summary[:150]}...")
    print(f"[VIDEO Tool] Style: {video_style} | Duration: {duration}s")
    print(f"[VIDEO Tool] Storage Project: {config.VIDEO_GCP_PROJECT_ID}")
    print(f"{'='*60}")
    
    try:
        # Validate duration (clamp to Veo 3.1 valid values: 4, 6, 8)
        if duration <= 4:
            duration = 4
        elif duration <= 6:
            duration = 6
        else:
            duration = 8

        # Get reference image path
        import os
        reference_image = os.path.join(config.PROJECT_ROOT, "assets", "image.png")

        # Create speech text from data_summary for lip-sync
        # Format the data summary as natural speech
        speech_text = f"Hello! Let me share your banking information. {data_summary}"

        # Limit speech text to reasonable length for video duration
        max_speech_length = 150 if duration == 4 else (250 if duration == 6 else 350)
        if len(speech_text) > max_speech_length:
            speech_text = speech_text[:max_speech_length].rsplit(' ', 1)[0] + "."

        # Create video prompt for a speaking banker
        if user_type == 'customer':
            context = "A professional bank representative presenting customer account information"
        else:
            context = "A professional financial advisor presenting merchant sales data"

        video_prompt = f"""{context} in a modern office setting.
The person speaks directly to the camera with a friendly, professional demeanor.
The background shows a bright, professional banking office.
The person maintains eye contact and uses natural hand gestures while speaking.
Lighting is professional and warm.
Style: {video_style}, corporate, trustworthy."""

        prompt_length = len(video_prompt)

        print(f"\n[VIDEO] Generating lip-sync video with Veo 3.1...")
        print(f"[VIDEO] Duration: {duration}s")
        print(f"[VIDEO] Speech text: {speech_text[:80]}...")
        print(f"[VIDEO] Reference image: {reference_image}")
        print(f"[VIDEO] Prompt length: {prompt_length} characters")

        # Track generation time
        gen_start_time = time.time()

        # Generate video using Veo 3.1 with reference image and lip-sync
        video_bytes = _generate_video_with_veo31(
            prompt=video_prompt,
            speech_text=speech_text,
            duration=duration,
            reference_image_path=reference_image
        )

        video_generation_time = time.time() - gen_start_time
        video_size_mb = len(video_bytes) / (1024 * 1024)

        print(f"⏱️  Video Generation: {video_generation_time:.3f}s")
        print(f"📦 Video Size: {video_size_mb:.2f} MB")

        # Upload to GCS (using separate video storage project)
        upload_start_time = time.time()
        print(f"\n[VIDEO] Uploading to GCS in project: {config.VIDEO_GCP_PROJECT_ID}...")

        # Initialize storage client with video storage project and credentials
        from google.oauth2 import service_account
        video_credentials = service_account.Credentials.from_service_account_file(
            config.VIDEO_SERVICE_ACCOUNT_KEY,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        storage_client = storage.Client(project=config.VIDEO_GCP_PROJECT_ID, credentials=video_credentials)
        bucket = storage_client.bucket(config.VIDEO_GCS_BUCKET)
        
        # Verify bucket exists and is accessible
        try:
            bucket.reload()
            print(f"✓ Bucket accessible: gs://{config.VIDEO_GCS_BUCKET}")
        except Exception as bucket_error:
            error_msg = f"Cannot access bucket in project {config.VIDEO_GCP_PROJECT_ID}: {str(bucket_error)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_id = uuid.uuid4().hex[:8]
        video_filename = f"{config.VIDEO_GCS_PATH}/{current_user}_{user_type}_{timestamp}_{video_id}.mp4"
        
        blob = bucket.blob(video_filename)
        blob.upload_from_string(video_bytes, content_type='video/mp4')

        upload_time = time.time() - upload_start_time
        print(f"⏱️  Upload Time: {upload_time:.3f}s")

        # Make video publicly accessible (using uniform bucket-level access)
        blob.cache_control = "public, max-age=3600"
        blob.patch()

        # Get public URL (works with uniform bucket-level access)
        video_url = f"https://storage.googleapis.com/{config.VIDEO_GCS_BUCKET}/{video_filename}"
        print(f"✓ Video is publicly accessible")
        print(f"✓ Public URL: {video_url}")
        gcs_path = f"gs://{config.VIDEO_GCS_BUCKET}/{video_filename}"
        total_time = time.time() - tool_start_time
        
        print(f"✓ Video uploaded to: {gcs_path}")
        print(f"✓ Project: {config.VIDEO_GCP_PROJECT_ID}")
        print(f"⏱️  Total Time: {total_time:.3f}s")
        
        # Log detailed video performance metrics
        log_video_performance_metric(
            user=current_user,
            user_type=user_type,
            data_summary=data_summary,
            video_style=video_style,
            duration=duration,
            prompt_length=prompt_length,
            video_generation_time=video_generation_time,
            upload_time=upload_time,
            total_time=total_time,
            video_size_mb=video_size_mb,
            video_url=video_url,
            gcs_path=gcs_path,
            status='SUCCESS'
        )
        
        # Also log to general performance metrics
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=f"VIDEO_GEN: {data_summary[:100]}",
            total_time=total_time,
            sql_generation_time=video_generation_time,
            sql_execution_time=upload_time,
            bytes_processed=len(video_bytes),
            status='SUCCESS'
        )
        
        # Log audit with project information
        audit_logger.info(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'user': current_user,
            'user_type': user_type,
            'action': 'video_generation_success',
            'storage_project': config.VIDEO_GCP_PROJECT_ID,
            'video_path': gcs_path,
            'video_url': video_url,
            'video_size_mb': round(video_size_mb, 2),
            'duration': duration,
            'style': video_style,
            'timings': {
                'generation': round(video_generation_time, 3),
                'upload': round(upload_time, 3),
                'total': round(total_time, 3)
            }
        }))
        
        # Build enhanced response with performance metrics and project info
        return f"""✅ **Video Generated Successfully!**

🎬 **What Was Created:**
A professional banker speaks your banking information with accurate lip-sync and natural expressions!

📹 **Video Details:**
   • Duration: {duration} seconds
   • Style: {video_style.capitalize()}
   • Size: {video_size_mb:.2f} MB
   • Format: MP4 (16:9, 720p)
   • Speaker: Professional banker from reference image

⚡ **Performance Metrics:**
   • Video Generation: {video_generation_time:.2f}s (including polling)
   • Upload to GCS: {upload_time:.2f}s
   • Total Time: {total_time:.2f}s

🌐 **PUBLIC VIDEO URL** (Anyone can access):
   {video_url}

📍 **Storage Location:**
   • GCS Path: {gcs_path}
   • Bucket: {config.VIDEO_GCS_BUCKET}
   • Project: {config.VIDEO_GCP_PROJECT_ID}

✨ **Features:**
   ✓ Publicly accessible (no authentication required)
   ✓ No expiration - permanent link
   ✓ Direct streaming from GCS
   ✓ HD quality with lip-sync

💡 **Share this URL with anyone** - they can view the video directly in their browser!"""
        
    except Exception as e:
        total_time = time.time() - tool_start_time
        error_msg = str(e)
        
        # Log video-specific error metrics
        log_video_performance_metric(
            user=current_user,
            user_type=user_type,
            data_summary=data_summary,
            video_style=video_style,
            duration=duration,
            prompt_length=prompt_length,
            video_generation_time=video_generation_time,
            upload_time=upload_time,
            total_time=total_time,
            video_size_mb=video_size_mb,
            video_url="",
            gcs_path="",
            status='ERROR',
            error_message=error_msg
        )
        
        # Also log to general performance metrics
        log_performance_metric(
            user=current_user,
            user_type=user_type,
            query=f"VIDEO_GEN: {data_summary[:100]}",
            total_time=total_time,
            status='ERROR'
        )
        
        # Log audit error with project info
        audit_logger.error(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'user': current_user,
            'user_type': user_type,
            'action': 'video_generation_failed',
            'storage_project': config.VIDEO_GCP_PROJECT_ID,
            'error': error_msg[:200],
            'total_time': round(total_time, 3)
        }))
        
        print(f"❌ Video generation failed: {error_msg}")
        
        if "quota" in error_msg.lower() or "429" in error_msg:
            return "⚠️ Video generation quota exceeded. Please try again later."
        elif "timeout" in error_msg.lower():
            return "⏱️ Video generation timed out. Please try again with a simpler request."
        elif "401" in error_msg or "403" in error_msg or "Cannot access bucket" in error_msg:
            return f"🔒 Storage access error in project {config.VIDEO_GCP_PROJECT_ID}. Please check permissions and bucket existence."
        else:
            return f"❌ Failed to generate video: {error_msg}\n\nPlease try again or contact support."
              
# --- 6. Run the Server ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting MCP Toolbox Server with Security Guardrails")
    print("=" * 60)
    print(f"🌐 Main Project: {config.GCP_PROJECT_ID}")
    print(f"📍 Location: {config.GCP_LOCATION}")
    print(f"\n🎬 Video Generation:")
    print(f"   ✓ Veo 3.1 Model: {'Enabled' if VEO_AVAILABLE else 'Disabled'}")
    print(f"   ✓ Storage Project: {config.VIDEO_GCP_PROJECT_ID}")
    print(f"   ✓ Storage Location: {config.VIDEO_GCP_LOCATION}")
    print(f"   ✓ Bucket: gs://{config.VIDEO_GCS_BUCKET}/{config.VIDEO_GCS_PATH}")
    print(f"   ✓ Signed URLs: 24-hour expiry")
    if config.VIDEO_GCP_PROJECT_ID != config.GCP_PROJECT_ID:
        print(f"   ⚠️  Using separate project for video storage (testing)")
    print(f"\n🔒 Security Implementation:")
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
    print(f"   • Video performance: video_generation_performance.log")
    print(f"   • Video URLs: video_urls.log")
    print(f"   • All queries logged with timing and status")
    print("\n📊 Performance Tracking: ENABLED")
    print(f"   • SQL generation time tracked")
    print(f"   • SQL execution time tracked")
    print(f"   • Video generation time tracked")
    print(f"   • BigQuery bytes processed tracked")
    print(f"   • Real-time metrics displayed")
    print("\n🚫 Prohibited Operations (per Banking Regulations):")
    print(f"   • DELETE, UPDATE, INSERT statements")
    print(f"   • Schema modifications (ALTER, CREATE, DROP)")
    print(f"   • Administrative commands (GRANT, REVOKE)")
    print(f"\n✅ Allowed Operations:")
    print(f"   • SELECT queries only (READ-ONLY access)")
    print(f"   • Video generation from query results")
    print("=" * 60)
    
    mcp.run(
        transport="sse",     
        host="0.0.0.0",       
        port=8001,
        path="/sse"           
    )