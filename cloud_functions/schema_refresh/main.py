"""
Cloud Function for scheduled schema refresh and embedding re-indexing
Triggers weekly to refresh both table and PDF embeddings in BigQuery
"""

import functions_framework
import os
import json
from datetime import datetime
from google.cloud import bigquery

# Import indexers
from table_indexer import create_table_embeddings_internal
from pdf_indexer import create_pdf_embeddings_internal
from website_indexer import create_website_embeddings_internal


@functions_framework.http
def refresh_schema(request):
    """
    HTTP Cloud Function to refresh schema and re-index both table and PDF embeddings.

    Triggered by Cloud Scheduler weekly.

    Returns:
        JSON response with status and details
    """
    start_time = datetime.utcnow()

    print(f"{'='*60}")
    print(f"🔄 SCHEDULED REFRESH STARTED")
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
        print("\n[1/4] Initializing GCP clients...")
        bq_client = bigquery.Client(project=project_id)
        print("✓ Clients initialized")
        results['steps']['init'] = 'success'

        # Step 2: Re-index tables to BigQuery
        print("\n[2/4] Re-indexing table embeddings to BigQuery...")
        table_result = create_table_embeddings_internal(
            project_id=project_id,
            dataset=dataset,
            location=location
        )

        if table_result['success']:
            print(f"✓ Indexed {table_result['tables_indexed']} tables")
            results['steps']['table_indexing'] = {
                'status': 'success',
                'tables_indexed': table_result['tables_indexed']
            }
        else:
            print(f"⚠️  Table indexing failed")
            results['steps']['table_indexing'] = {
                'status': 'error',
                'message': table_result.get('error', 'Unknown error')
            }

        # Step 3: Re-index PDF to BigQuery
        print("\n[3/4] Re-indexing PDF embeddings to BigQuery...")
        pdf_result = create_pdf_embeddings_internal(
            project_id=project_id,
            dataset=dataset,
            location=location
        )

        if pdf_result['success']:
            if pdf_result.get('skipped'):
                print(f"ℹ️  PDF indexing skipped: {pdf_result.get('message')}")
                results['steps']['pdf_indexing'] = {
                    'status': 'skipped',
                    'message': pdf_result.get('message')
                }
            else:
                print(f"✓ Indexed {pdf_result['chunks_indexed']} PDF chunks")
                results['steps']['pdf_indexing'] = {
                    'status': 'success',
                    'chunks_indexed': pdf_result['chunks_indexed']
                }
        else:
            print(f"⚠️  PDF indexing failed")
            results['steps']['pdf_indexing'] = {
                'status': 'error',
                'message': pdf_result.get('error', 'Unknown error')
            }

        # Step 4: Re-index websites to BigQuery
        print("\n[4/4] Re-indexing website embeddings to BigQuery...")
        website_result = create_website_embeddings_internal(
            project_id=project_id,
            dataset=dataset,
            location=location
        )

        if website_result['success']:
            if website_result.get('skipped'):
                print(f"ℹ️  Website indexing skipped: {website_result.get('message')}")
                results['steps']['website_indexing'] = {
                    'status': 'skipped',
                    'message': website_result.get('message')
                }
            else:
                print(f"✓ Indexed {website_result['chunks_indexed']} website chunks from {website_result.get('websites_scraped', 0)} websites")
                results['steps']['website_indexing'] = {
                    'status': 'success',
                    'chunks_indexed': website_result['chunks_indexed'],
                    'websites_scraped': website_result.get('websites_scraped', 0)
                }
        else:
            print(f"⚠️  Website indexing failed")
            results['steps']['website_indexing'] = {
                'status': 'error',
                'message': website_result.get('error', 'Unknown error')
            }

        # Calculate duration
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration
        results['status'] = 'success'

        print(f"\n{'='*60}")
        print(f"✅ REFRESH COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Tables indexed: {table_result.get('tables_indexed', 0)}")
        print(f"PDF chunks indexed: {pdf_result.get('chunks_indexed', 0)}")
        print(f"Website chunks indexed: {website_result.get('chunks_indexed', 0)}")
        print(f"{'='*60}\n")

        return results, 200

    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*60}")
        print(f"❌ REFRESH FAILED")
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
