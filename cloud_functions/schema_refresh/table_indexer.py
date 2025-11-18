"""
Table indexer module for Cloud Function
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from bigquery_vector_store import BigQueryVectorStore
from schema_cache_manager import fetch_bigquery_schema, generate_all_table_contexts


def create_table_embeddings_internal(
    project_id: str,
    dataset: str,
    location: str = "us-central1"
) -> dict:
    """
    Create and store table embeddings in BigQuery.

    This is an internal function meant to be called by Cloud Functions.

    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        location: GCP location

    Returns:
        Dict with success status and details
    """
    try:
        # Step 1: Fetch schema from BigQuery
        print("  Fetching schema from BigQuery...")
        bq_client = bigquery.Client(project=project_id)
        schema_info = fetch_bigquery_schema(bq_client, dataset)
        print(f"  ✓ Fetched {len(schema_info)} tables")

        # Step 2: Generate contexts with Gemini
        print("  Generating table contexts with Gemini...")
        llm = ChatVertexAI(
            model="gemini-pro",
            project=project_id,
            location=location
        )
        table_contexts = generate_all_table_contexts(schema_info, llm)
        print(f"  ✓ Generated {len(table_contexts)} contexts")

        # Step 3: Create embeddings
        print("  Initializing Vertex AI embeddings...")
        embeddings_model = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id,
            location=location
        )

        print("  Creating embeddings for table schemas...")
        embeddings_data = []

        for table_name, info in schema_info.items():
            context = table_contexts.get(table_name, {})

            # Create comprehensive text representation
            fields_text = "\n".join([
                f"  - {field['name']} ({field['type']}) {'[REQUIRED]' if field['mode'] == 'REQUIRED' else ''}"
                for field in info.get("fields", [])
            ])

            content = f"""
TABLE: {table_name}

DESCRIPTION: {context.get('description', 'N/A')}

BUSINESS CONTEXT: {context.get('business_context', 'N/A')}

USAGE: {context.get('usage', 'N/A')}

COMMON QUERIES: {context.get('common_queries', 'N/A')}

COLUMNS:
{fields_text}

ROW COUNT: {info.get('num_rows', 0):,}

SENSITIVE DATA: {'Yes' if context.get('sensitive', False) else 'No'}
"""

            # Generate embedding
            embedding = embeddings_model.embed_query(content.strip())

            embeddings_data.append({
                "id": f"table_{table_name}",
                "source_name": table_name,
                "content": content.strip(),
                "embedding": embedding,
                "metadata": {
                    "num_rows": info.get("num_rows", 0),
                    "num_columns": len(info.get("fields", [])),
                    "description": context.get('description', ''),
                    "sensitive": context.get('sensitive', False)
                }
            })

            print(f"    ✓ {table_name}")

        # Initialize BigQuery Vector Store
        print("  Initializing BigQuery Vector Store...")
        vector_store = BigQueryVectorStore(dataset_name=dataset)
        vector_store.create_embeddings_table()

        # Insert embeddings
        print("  Inserting embeddings into BigQuery...")
        success = vector_store.insert_embeddings(
            embeddings_data=embeddings_data,
            source_type="table"
        )

        if success:
            return {
                'success': True,
                'tables_indexed': len(embeddings_data),
                'embedding_dimensions': len(embeddings_data[0]['embedding']) if embeddings_data else 0
            }
        else:
            return {
                'success': False,
                'message': 'Failed to insert embeddings'
            }

    except Exception as e:
        print(f"  ❌ Error in table indexing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }
