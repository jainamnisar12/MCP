"""
Simplified config for Cloud Function
Gets configuration from environment variables set during deployment
"""

import os

# GCP Configuration (set via Cloud Function environment variables)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET")

# These are not needed for Cloud Function but kept for compatibility
PROJECT_ROOT = "/workspace"
PDF_PATH = None
VECTOR_STORE_PATH = None
VECTOR_STORE_PATH_OPENAI = None
OPENAI_API_KEY = None
GOOGLE_API_KEY = None
