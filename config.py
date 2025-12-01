import os
from dotenv import load_dotenv

# --- 1. Define Paths ---
# Get the project root directory (where this config.py file is)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 2. Load .env file from Project Root ---
# This looks for .env in the 'MCP' folder
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Set gRPC DNS resolver to native (fixes C-ares DNS issues)
if not os.getenv("GRPC_DNS_RESOLVER"):
    os.environ["GRPC_DNS_RESOLVER"] = "native"

# --- 3. Google Cloud Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")  # Add this line with default
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET")

# Get the relative path from the .env file
SERVICE_ACCOUNT_KEY_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if SERVICE_ACCOUNT_KEY_FILE:
    # Build the full, absolute path to the key file
    abs_service_path = os.path.join(PROJECT_ROOT, SERVICE_ACCOUNT_KEY_FILE)
    if os.path.exists(abs_service_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_service_path
    else:
        print(f"Warning: Service account key not found at {abs_service_path}")
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set in .env")


# --- 4. API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # For agent.py

# --- 5. File Paths (Absolute) ---
# Builds full path to 'MCP/data/...'
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "UPI Transaction Process Explained.pdf")
# Builds full path to 'MCP/vector_store_openai'
VECTOR_STORE_PATH_OPENAI = os.path.join(PROJECT_ROOT, "vector_store_openai")
# Builds full path to 'MCP/vector_store_gemini'
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store_gemini")
# Website vector store paths
WEBSITE_VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store_websites")
COMBINED_VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store_combined")

# --- 6. Website URLs to Index ---
WEBSITE_URLS = [
    # Add your website URLs here
    "https://www.mindgate.solutions/retail-payments/",
    "https://www.mindgate.solutions/transaction-banking-platform/",
    "https://www.mindgate.solutions/government-solutions/",
    "https://www.mindgate.solutions/central-infrastructure/",
    "https://www.mindgate.solutions/reconciliation-settlement/",
    "https://www.mindgate.solutions/merchants-tpap-offerings/"
]

# GCS_VIDEO_BUCKET = os.getenv("GCS_VIDEO_BUCKET")

# VIDEO_DEFAULT_DURATION = 8  # seconds
# VIDEO_STORAGE_PATH = "videos/"

# Video Generation Configuration (Using Main Project)
VIDEO_GCP_PROJECT_ID = GCP_PROJECT_ID  # Use same project as main
VIDEO_GCP_LOCATION = GCP_LOCATION  # Same location as main project
VIDEO_GCS_BUCKET = "mcp_chatbot"  # Video storage bucket
VIDEO_GCS_PATH = "videos"  # Folder path within bucket

# Use main service account for video operations
VIDEO_SERVICE_ACCOUNT_KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# --- 7. Validation ---
if not all([GCP_PROJECT_ID, BIGQUERY_DATASET, GOOGLE_API_KEY]):
    raise ValueError(
        "Missing required environment variables from .env file:\n"
        "GCP_PROJECT_ID, BIGQUERY_DATASET, GOOGLE_API_KEY"
    )

if not os.path.exists(PDF_PATH):
     print(f"Warning: PDF_PATH not found at {PDF_PATH}")