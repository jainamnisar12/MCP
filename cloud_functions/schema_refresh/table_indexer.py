"""
Table indexer module for Cloud Function
Fetches schemas, generates contexts, and creates embeddings
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.prompts import PromptTemplate
from bigquery_vector_store import BigQueryVectorStore
from typing import Dict, List


def fetch_bigquery_schema(bq_client: bigquery.Client, dataset: str) -> Dict:
    """
    Fetch schema information for all tables in a dataset.

    Args:
        bq_client: BigQuery client
        dataset: Dataset name

    Returns:
        Dict with table schemas
    """
    print(f"  Fetching schema for dataset: {dataset}...")

    schema_info = {}

    try:
        tables = bq_client.list_tables(dataset)

        for table_ref in tables:
            table_id = table_ref.table_id

            # Skip embeddings table
            if table_id == "vector_embeddings":
                continue

            # Get full table details
            table = bq_client.get_table(f"{bq_client.project}.{dataset}.{table_id}")

            columns = []
            for field in table.schema:
                columns.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or ""
                })

            schema_info[table_id] = {
                "columns": columns,
                "description": table.description or "",
                "num_rows": table.num_rows or 0
            }

        print(f"  ✓ Fetched schema for {len(schema_info)} tables")
        return schema_info

    except Exception as e:
        print(f"  ❌ Error fetching schema: {e}")
        raise


def generate_table_context(table_name: str, table_info: Dict, llm=None) -> str:
    """
    Generate a semantic context for a table.

    Args:
        table_name: Name of the table
        table_info: Table schema information
        llm: ChatVertexAI instance (optional, will use basic description if None)

    Returns:
        Generated context string
    """
    # Generate basic context from schema
    num_cols = len(table_info['columns'])
    column_names = ", ".join([col['name'] for col in table_info['columns'][:5]])
    if num_cols > 5:
        column_names += f", and {num_cols - 5} more"

    basic_context = f"Table {table_name} with {num_cols} columns including {column_names}."

    # Add description if available
    if table_info.get('description'):
        basic_context += f" {table_info['description']}"

    # Try to use Gemini for enhanced context (but don't fail if it's not available)
    if llm:
        columns_text = "\n".join([
            f"  - {col['name']} ({col['type']}): {col.get('description', '')}"
            for col in table_info['columns']
        ])

        prompt_template = PromptTemplate(
            input_variables=["table_name", "columns", "description"],
            template="""Analyze this BigQuery table and provide a concise semantic description.

Table: {table_name}
Description: {description}
Columns:
{columns}

Provide a 2-3 sentence description that explains:
1. What this table stores
2. Key relationships or purposes
3. When it would be relevant for queries

Context:"""
        )

        prompt = prompt_template.format(
            table_name=table_name,
            columns=columns_text,
            description=table_info.get('description', 'No description')
        )

        try:
            response = llm.invoke(prompt)
            context = response.content.strip()
            return context
        except Exception as e:
            print(f"  ⚠️  Gemini context generation failed for {table_name}, using basic context")

    return basic_context


def generate_all_table_contexts(schema_info: Dict, llm) -> Dict[str, str]:
    """
    Generate contexts for all tables.

    Args:
        schema_info: Schema information for all tables
        llm: ChatVertexAI instance

    Returns:
        Dict mapping table names to contexts
    """
    print("  Generating table contexts with Gemini...")

    contexts = {}
    total_tables = len(schema_info)

    for idx, (table_name, table_info) in enumerate(schema_info.items(), 1):
        print(f"    [{idx}/{total_tables}] Generating context for {table_name}...")
        context = generate_table_context(table_name, table_info, llm)
        contexts[table_name] = context

    print(f"  ✓ Generated contexts for {len(contexts)} tables")
    return contexts


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
        print("  Initializing BigQuery client...")
        bq_client = bigquery.Client(project=project_id)

        schema_info = fetch_bigquery_schema(bq_client, dataset)

        if not schema_info:
            return {
                'success': False,
                'error': 'No tables found in dataset'
            }

        # Step 2: Generate contexts (try Gemini, fallback to basic)
        llm = None
        try:
            print("  Initializing Gemini for context generation...")
            llm = ChatVertexAI(
                model="gemini-1.5-flash-001",
                project=project_id,
                location=location,
                temperature=0.0
            )
            print("  ✓ Gemini initialized")
        except Exception as e:
            print(f"  ⚠️  Gemini not available, will use basic context generation: {e}")

        table_contexts = generate_all_table_contexts(schema_info, llm)

        # Step 3: Create embeddings
        print("  Initializing Vertex AI embeddings...")
        embeddings_model = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id,
            location=location
        )

        print("  Creating embeddings for tables...")
        embeddings_data = []

        for table_name, context in table_contexts.items():
            table_info = schema_info[table_name]

            # Create content to embed
            columns_summary = ", ".join([col['name'] for col in table_info['columns'][:10]])
            if len(table_info['columns']) > 10:
                columns_summary += "..."

            content = f"""Table: {table_name}

Context: {context}

Columns: {columns_summary}

Schema Details:
{chr(10).join([f"- {col['name']} ({col['type']})" for col in table_info['columns']])}"""

            # Generate embedding
            embedding = embeddings_model.embed_query(content.strip())

            # Prepare metadata - convert complex types to JSON-compatible format
            metadata = {
                "table_name": table_name,
                "num_columns": len(table_info['columns']),
                "num_rows": int(table_info.get('num_rows', 0)),
                "context": context,
                "column_names": ", ".join([col['name'] for col in table_info['columns']])  # Store as string instead of array
            }

            embeddings_data.append({
                "id": f"table_{table_name}",
                "source_name": table_name,
                "content": content.strip(),
                "embedding": embedding,
                "metadata": metadata
            })

            print(f"    ✓ Created embedding for {table_name}")

        print(f"  ✓ Created {len(embeddings_data)} table embeddings")

        # Step 4: Initialize BigQuery Vector Store and insert
        print("  Initializing BigQuery Vector Store...")
        vector_store = BigQueryVectorStore(dataset_name=dataset)
        vector_store.create_embeddings_table()

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
                'error': 'Failed to insert embeddings'
            }

    except Exception as e:
        print(f"  ❌ Error in table indexing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }
