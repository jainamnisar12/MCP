"""
Table Retriever - Retrieves relevant tables using BigQuery vector search
"""

import os
import sys
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from schema_cache_manager import SchemaCache
from .bigquery_vector_store import BigQueryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings


class TableRetriever:
    """Retrieves relevant tables using semantic search from BigQuery."""

    def __init__(self):
        self.vector_store = None
        self.embeddings_model = None
        self.schema_cache = SchemaCache()

    def initialize(self) -> bool:
        """Initialize the vector store and embeddings."""
        try:
            # Initialize embeddings model
            self.embeddings_model = VertexAIEmbeddings(
                model_name="text-embedding-004",
                project=config.GCP_PROJECT_ID,
                location=config.GCP_LOCATION if hasattr(config, 'GCP_LOCATION') else "us-central1"
            )

            # Initialize BigQuery vector store
            self.vector_store = BigQueryVectorStore()

            # Verify table exists
            stats = self.vector_store.get_stats()
            if 'table' not in stats or stats['table'] == 0:
                print(f"⚠️  No table embeddings found in BigQuery. Please run the Cloud Function to populate embeddings.")
                return False

            print(f"✓ Found {stats['table']} table embeddings in BigQuery")
            return True

        except Exception as e:
            print(f"⚠️  Could not initialize table retriever: {e}")
            return False

    def retrieve_relevant_tables(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve the most relevant tables for a given query using semantic search.

        Args:
            query: The user's natural language query
            k: Number of top tables to retrieve (default: 3)
            score_threshold: Minimum similarity score threshold (default: 0.5)

        Returns:
            List of dictionaries containing table information and relevance scores
        """
        if not self.vector_store or not self.embeddings_model:
            if not self.initialize():
                return self._get_fallback_tables()

        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_model.embed_query(query)

            # Search for similar tables in BigQuery
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                source_type="table",
                top_k=k,
                min_similarity=score_threshold
            )

            if not results:
                print(f"[RAG] No tables found above threshold {score_threshold}, using fallback")
                return self._get_fallback_tables()

            # Format results
            relevant_tables = []
            for result in results:
                relevant_tables.append({
                    "table_name": result["source_name"],
                    "description": result["metadata"].get("description", ""),
                    "num_rows": result["metadata"].get("num_rows", 0),
                    "num_columns": result["metadata"].get("num_columns", 0),
                    "sensitive": result["metadata"].get("sensitive", False),
                    "similarity_score": round(result["similarity"], 3),
                    "content": result["content"]
                })

            return relevant_tables

        except Exception as e:
            print(f"⚠️  Error during table retrieval: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_tables()

    def _get_fallback_tables(self) -> List[Dict]:
        """Fallback to UPI tables if BigQuery search fails."""
        cached_data = self.schema_cache.load_cache()

        if not cached_data:
            return []

        schema_info = cached_data.get("schema_info", {})
        table_contexts = cached_data.get("table_contexts", {})

        # Return UPI tables as fallback
        fallback_tables = []
        for table_name in ["upi_transaction", "upi_customer", "upi_merchant", "upi_bank"]:
            if table_name in schema_info:
                info = schema_info[table_name]
                context = table_contexts.get(table_name, {})

                fallback_tables.append({
                    "table_name": table_name,
                    "description": context.get("description", ""),
                    "num_rows": info.get("num_rows", 0),
                    "num_columns": len(info.get("fields", [])),
                    "sensitive": context.get("sensitive", False),
                    "similarity_score": 0.0,
                    "content": f"Fallback table: {table_name}"
                })

        return fallback_tables

    def get_table_schema_text(self, table_names: List[str]) -> str:
        """
        Get formatted schema text for specific tables.

        Args:
            table_names: List of table names to include

        Returns:
            Formatted schema string for the LLM
        """
        cached_data = self.schema_cache.load_cache()

        if not cached_data:
            return "No schema available."

        schema_info = cached_data.get("schema_info", {})
        table_contexts = cached_data.get("table_contexts", {})

        schema_parts = []
        schema_parts.append("=" * 60)
        schema_parts.append("RELEVANT DATABASE TABLES (Retrieved via RAG from BigQuery)")
        schema_parts.append("=" * 60)

        for table_name in table_names:
            if table_name not in schema_info:
                continue

            info = schema_info[table_name]
            context = table_contexts.get(table_name, {})

            schema_parts.append(f"\n📊 TABLE: {table_name.upper()}")
            schema_parts.append("-" * 60)

            if context.get("description"):
                schema_parts.append(f"📝 Description: {context['description']}")

            if context.get("business_context"):
                schema_parts.append(f"💼 Business Context: {context['business_context']}")

            if context.get("usage"):
                schema_parts.append(f"🎯 Usage: {context['usage']}")

            schema_parts.append(f"📊 Rows: {info['num_rows']:,}")

            if context.get("sensitive"):
                schema_parts.append(f"🔒 Sensitive Data: Yes - Row-level security enforced")

            schema_parts.append("\n📋 Columns:")

            for field in info.get("fields", []):
                field_desc = f"  • {field['name']} ({field['type']})"
                if field['mode'] == 'REQUIRED':
                    field_desc += " [REQUIRED]"
                if field.get('description'):
                    field_desc += f" - {field['description']}"
                schema_parts.append(field_desc)

            schema_parts.append("")  # Empty line between tables

        schema_parts.append("=" * 60)

        return "\n".join(schema_parts)


# Utility function for easy usage
def get_relevant_tables_for_query(query: str, k: int = 3) -> List[Dict]:
    """
    Convenience function to retrieve relevant tables for a query.

    Args:
        query: User's natural language query
        k: Number of tables to retrieve

    Returns:
        List of relevant table dictionaries
    """
    retriever = TableRetriever()
    retriever.initialize()
    return retriever.retrieve_relevant_tables(query, k=k)
