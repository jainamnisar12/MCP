"""
Cloud Function for scheduled schema refresh and embedding re-indexing
Triggered by Cloud Scheduler to run weekly
"""

import functions_framework
import os
import json
from datetime import datetime
from google.cloud import bigquery
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# Import table indexer
from table_indexer import create_table_embeddings_internal


@functions_framework.http
def refresh_schema(request):
    """
    HTTP Cloud Function to refresh schema and re-index embeddings.

    Triggered by Cloud Scheduler weekly.

    Returns:
        JSON response with status and details
    """
    start_time = datetime.utcnow()

    print(f"{'='*60}")
    print(f"🔄 SCHEDULED SCHEMA REFRESH STARTED")
    print(f"{'='*60}")
    print(f"Timestamp: {start_time.isoformat()}")
    print(f"{'='*60}\n")

    # Get configuration from environment variables
    project_id = os.environ.get('GCP_PROJECT_ID')
    dataset = os.environ.get('BIGQUERY_DATASET')
    location = os.environ.get('GCP_LOCATION', 'us-central1')

    if not all([project_id, dataset]):
        error_msg = "Missing required environment variables: GCP_PROJECT_ID, BIGQUERY_DATASET"
        print(f"❌ {error_msg}")
        return {
            'status': 'error',
            'message': error_msg
        }, 500

    results = {
        'start_time': start_time.isoformat(),
        'project_id': project_id,
        'dataset': dataset,
        'steps': {}
    }

    try:
        # Step 1: Initialize clients
        print("\n[1/2] Initializing GCP clients...")
        bq_client = bigquery.Client(project=project_id)
        print("✓ Clients initialized")
        results['steps']['init'] = 'success'

        # Step 2: Re-index tables to BigQuery
        # This will fetch schema internally and re-index
        print("\n[2/2] Re-indexing table embeddings to BigQuery...")
        indexing_result = create_table_embeddings_internal(
            project_id=project_id,
            dataset=dataset,
            location=location
        )

        if indexing_result['success']:
            print(f"✓ Indexed {indexing_result['tables_indexed']} tables")
            results['steps']['indexing'] = {
                'status': 'success',
                'tables_indexed': indexing_result['tables_indexed']
            }
        else:
            print(f"⚠️  Indexing failed")
            results['steps']['indexing'] = {
                'status': 'error',
                'message': indexing_result.get('error', 'Unknown error')
            }

        # Calculate duration
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration
        results['status'] = 'success'

        print(f"\n{'='*60}")
        print(f"✅ SCHEMA REFRESH COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Duration: {duration:.2f} seconds")
        tables_count = results['steps'].get('indexing', {}).get('tables_indexed', 0)
        print(f"Tables refreshed: {tables_count}")
        print(f"{'='*60}\n")

        return results, 200

    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*60}")
        print(f"❌ SCHEMA REFRESH FAILED")
        print(f"{'='*60}")
        print(f"Error: {error_msg}")
        print(f"{'='*60}\n")

        import traceback
        traceback.print_exc()

        results['status'] = 'error'
        results['error'] = error_msg
        results['end_time'] = datetime.utcnow().isoformat()

        return results, 500


# For local testing
if __name__ == '__main__':
    class MockRequest:
        pass

    result, status_code = refresh_schema(MockRequest())
    print(f"\nResult: {json.dumps(result, indent=2)}")
    print(f"Status: {status_code}")
