"""
Table Schema Indexer - Creates embeddings for table schemas and stores in BigQuery
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from schema_cache_manager import SchemaCache
from bigquery_vector_store import BigQueryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings


def create_table_embeddings():
    """Create embeddings for table schemas and store in BigQuery."""

    print("\n" + "="*60)
    print("📊 TABLE SCHEMA INDEXING TO BIGQUERY")
    print("="*60)

    # Step 1: Load table schemas
    print("\n[1/5] Loading table schemas from cache...")
    try:
        cache = SchemaCache()
        cached_data = cache.load_cache()

        if not cached_data:
            print("❌ Error: No schema cache found. Please run the MCP server first to generate schema cache.")
            return False

        schema_info = cached_data.get("schema_info", {})
        table_contexts = cached_data.get("table_contexts", {})

        print(f"✓ Loaded {len(schema_info)} tables from cache")
    except Exception as e:
        print(f"❌ Error loading schema cache: {e}")
        return False

    # Step 2: Initialize Vertex AI embeddings
    print("\n[2/5] Initializing Vertex AI embeddings...")
    try:
        embeddings_model = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION if hasattr(config, 'GCP_LOCATION') else "us-central1"
        )
        print("✓ Vertex AI embeddings initialized")
    except Exception as e:
        print(f"❌ Error initializing Vertex AI embeddings: {e}")
        return False

    # Step 3: Create embeddings data
    print("\n[3/5] Creating embeddings for table schemas...")
    embeddings_data = []

    for table_name, info in schema_info.items():
        context = table_contexts.get(table_name, {})

        # Create a comprehensive text representation of the table
        fields_text = "\n".join([
            f"  - {field['name']} ({field['type']}) {'[REQUIRED]' if field['mode'] == 'REQUIRED' else ''}"
            for field in info.get("fields", [])
        ])

        # Build rich content for embedding
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
        try:
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

            print(f"  ✓ Created embedding for: {table_name}")

        except Exception as e:
            print(f"  ❌ Error creating embedding for {table_name}: {e}")
            continue

    print(f"✓ Created {len(embeddings_data)} table embeddings")

    # Step 4: Initialize BigQuery Vector Store
    print("\n[4/5] Initializing BigQuery Vector Store...")
    try:
        vector_store = BigQueryVectorStore()
        vector_store.create_embeddings_table()
        print("✓ BigQuery vector store ready")
    except Exception as e:
        print(f"❌ Error initializing BigQuery vector store: {e}")
        return False

    # Step 5: Insert embeddings into BigQuery
    print("\n[5/5] Inserting embeddings into BigQuery...")
    try:
        success = vector_store.insert_embeddings(
            embeddings_data=embeddings_data,
            source_type="table"
        )

        if success:
            print("\n" + "="*60)
            print("✅ TABLE EMBEDDINGS STORED IN BIGQUERY!")
            print("="*60)
            print(f"📊 Total tables indexed: {len(embeddings_data)}")
            print(f"🗄️  Dataset: {config.BIGQUERY_DATASET}")
            print(f"📋 Table: vector_embeddings")
            print(f"🔧 Embedding model: text-embedding-004 (Vertex AI)")
            print(f"📐 Embedding dimensions: {len(embeddings_data[0]['embedding']) if embeddings_data else 'N/A'}")
            print("="*60 + "\n")
            return True
        else:
            print("❌ Failed to insert embeddings")
            return False

    except Exception as e:
        print(f"❌ Error inserting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_table_embeddings()
    sys.exit(0 if success else 1)
