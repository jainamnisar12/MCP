"""
Simplified configuration for Cloud Functions.
Reads configuration from environment variables instead of .env files.
"""

import os


# GCP Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# BigQuery Configuration
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET")

# PDF Configuration (optional)
SOURCE_PDF_PATH = os.getenv("SOURCE_PDF_PATH", "/workspace/data/UPI Transaction Process Explained.pdf")


def validate_config():
    """
    Validate that required environment variables are set.
    Raises ValueError if any required variables are missing.
    """
    required_vars = {
        "GCP_PROJECT_ID": GCP_PROJECT_ID,
        "BIGQUERY_DATASET": BIGQUERY_DATASET,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


# Validate on import (optional - can be removed if validation should be done elsewhere)
# validate_config()
