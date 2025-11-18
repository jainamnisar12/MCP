#!/bin/bash

# Cloud Function Deployment Script
# Deploys the schema refresh function to GCP

set -e  # Exit on error

echo "============================================================"
echo "🚀 DEPLOYING SCHEMA REFRESH CLOUD FUNCTION"
echo "============================================================"

# Configuration
FUNCTION_NAME="schema-refresh-weekly"
REGION="us-central1"
RUNTIME="python313"
ENTRY_POINT="refresh_schema"
TIMEOUT="540s"  # 9 minutes
MEMORY="1GB"    # Increased for Gemini + embeddings

# Get project ID and dataset from .env file
if [ -f "../../.env" ]; then
    # Load .env file (handles quoted values with spaces)
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z $key ]] && continue
        # Remove quotes from value if present
        value="${value%\"}"
        value="${value#\"}"
        # Export the variable
        export "$key=$value"
    done < <(grep -v '^#' ../../.env | grep -v '^$')
else
    echo "❌ Error: .env file not found in MCP directory"
    exit 1
fi

if [ -z "$GCP_PROJECT_ID" ] || [ -z "$BIGQUERY_DATASET" ]; then
    echo "❌ Error: GCP_PROJECT_ID and BIGQUERY_DATASET must be set in .env"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Project ID: $GCP_PROJECT_ID"
echo "  Dataset: $BIGQUERY_DATASET"
echo "  Region: $REGION"
echo "  Function: $FUNCTION_NAME"
echo ""

# Step 1: Copy necessary files
echo "[1/4] Copying necessary files..."
cp ../../bigquery_vector_store.py .
cp ../../schema_cache_manager.py .
echo "✓ Files copied"

# Step 2: Deploy Cloud Function
echo ""
echo "[2/4] Deploying Cloud Function..."
gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=$ENTRY_POINT \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=$TIMEOUT \
    --memory=$MEMORY \
    --set-env-vars GCP_PROJECT_ID=$GCP_PROJECT_ID,BIGQUERY_DATASET=$BIGQUERY_DATASET,GCP_LOCATION=$REGION \
    --service-account=$SERVICE_ACCOUNT_EMAIL

echo "✓ Cloud Function deployed"

# Get the function URL
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME --region=$REGION --gen2 --format='value(serviceConfig.uri)')

echo ""
echo "[3/4] Creating Cloud Scheduler job..."

# Create Cloud Scheduler job (runs every Monday at 2 AM)
gcloud scheduler jobs create http schema-refresh-weekly-job \
    --location=$REGION \
    --schedule="0 2 * * 1" \
    --uri=$FUNCTION_URL \
    --http-method=GET \
    --time-zone="America/New_York" \
    --description="Weekly schema refresh and embedding re-indexing" \
    --attempt-deadline=540s \
    || echo "⚠️  Scheduler job may already exist. Update it manually if needed."

echo "✓ Cloud Scheduler job configured"

# Step 4: Cleanup
echo ""
echo "[4/4] Cleaning up temporary files..."
rm -f bigquery_vector_store.py schema_cache_manager.py
echo "✓ Cleanup complete"

echo ""
echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "Function URL: $FUNCTION_URL"
echo ""
echo "Schedule: Every Monday at 2:00 AM (America/New_York)"
echo ""
echo "To test the function manually:"
echo "  curl $FUNCTION_URL"
echo ""
echo "To view logs:"
echo "  gcloud functions logs read $FUNCTION_NAME --region=$REGION --gen2"
echo ""
echo "To view scheduler jobs:"
echo "  gcloud scheduler jobs list --location=$REGION"
echo ""
echo "============================================================"
