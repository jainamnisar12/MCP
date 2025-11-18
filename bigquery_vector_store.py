"""
BigQuery Vector Store - Unified storage for embeddings in BigQuery
Supports both PDF and table embeddings with vector similarity search
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from google.cloud import bigquery
from datetime import datetime
import config


class BigQueryVectorStore:
    """Manages vector embeddings storage and retrieval in BigQuery."""

    def __init__(self, dataset_name: str = None, table_name: str = "vector_embeddings"):
        """
        Initialize BigQuery Vector Store.

        Args:
            dataset_name: BigQuery dataset name (defaults to config.BIGQUERY_DATASET)
            table_name: Table name for storing embeddings (default: vector_embeddings)
        """
        self.client = bigquery.Client(project=config.GCP_PROJECT_ID)
        self.dataset_name = dataset_name or config.BIGQUERY_DATASET
        self.table_name = table_name
        self.full_table_id = f"{config.GCP_PROJECT_ID}.{self.dataset_name}.{self.table_name}"

    def create_embeddings_table(self):
        """Create the embeddings table if it doesn't exist."""

        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED", description="Unique identifier"),
            bigquery.SchemaField("source_type", "STRING", mode="REQUIRED", description="Type: 'table' or 'pdf'"),
            bigquery.SchemaField("source_name", "STRING", mode="REQUIRED", description="Table name or PDF chunk ID"),
            bigquery.SchemaField("content", "STRING", mode="REQUIRED", description="Text content that was embedded"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED", description="Vector embedding (768 dimensions)"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE", description="Additional metadata"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED", description="Creation timestamp"),
        ]

        table = bigquery.Table(self.full_table_id, schema=schema)
        table.description = "Vector embeddings for RAG (PDF and table schemas)"

        try:
            table = self.client.create_table(table)
            print(f"✓ Created table {self.full_table_id}")
            return True
        except Exception as e:
            if "Already Exists" in str(e):
                print(f"✓ Table {self.full_table_id} already exists")
                return True
            else:
                print(f"❌ Error creating table: {e}")
                return False

    def insert_embeddings(
        self,
        embeddings_data: List[Dict],
        source_type: str
    ) -> bool:
        """
        Insert embeddings into BigQuery.

        Args:
            embeddings_data: List of dicts with keys: id, source_name, content, embedding, metadata
            source_type: 'table' or 'pdf'

        Returns:
            True if successful, False otherwise
        """
        print(f"\n[BigQuery] Inserting {len(embeddings_data)} {source_type} embeddings...")

        # First, delete existing embeddings for this source type to avoid duplicates
        delete_query = f"""
        DELETE FROM `{self.full_table_id}`
        WHERE source_type = '{source_type}'
        """

        try:
            self.client.query(delete_query).result()
            print(f"✓ Cleared existing {source_type} embeddings")
        except Exception as e:
            print(f"⚠️  Warning: Could not clear existing embeddings: {e}")

        # Prepare rows for insertion
        rows_to_insert = []
        for data in embeddings_data:
            row = {
                "id": data["id"],
                "source_type": source_type,
                "source_name": data["source_name"],
                "content": data["content"],
                "embedding": data["embedding"].tolist() if isinstance(data["embedding"], np.ndarray) else data["embedding"],
                "metadata": json.dumps(data.get("metadata", {})),
                "created_at": datetime.utcnow().isoformat()
            }
            rows_to_insert.append(row)

        # Insert in batches
        batch_size = 100
        total_inserted = 0

        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i + batch_size]

            try:
                errors = self.client.insert_rows_json(self.full_table_id, batch)

                if errors:
                    print(f"❌ Errors inserting batch {i//batch_size + 1}: {errors}")
                    return False
                else:
                    total_inserted += len(batch)
                    print(f"  ✓ Inserted batch {i//batch_size + 1}/{(len(rows_to_insert)-1)//batch_size + 1} ({len(batch)} rows)")

            except Exception as e:
                print(f"❌ Error inserting batch: {e}")
                return False

        print(f"✓ Successfully inserted {total_inserted} embeddings")
        return True

    def search_similar(
        self,
        query_embedding: List[float],
        source_type: str = None,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar embeddings using cosine similarity.

        Args:
            query_embedding: Query vector embedding
            source_type: Filter by source type ('table', 'pdf', or None for all)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of dicts with: id, source_name, content, similarity, metadata
        """

        # Convert query embedding to string format for SQL
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Build query with source type filter
        source_filter = f"AND source_type = '{source_type}'" if source_type else ""

        # BigQuery SQL for cosine similarity search
        query = f"""
        WITH query_embedding AS (
          SELECT {embedding_str} as query_vec
        ),
        similarities AS (
          SELECT
            id,
            source_type,
            source_name,
            content,
            metadata,
            embedding,
            -- Cosine similarity calculation
            (
              SELECT SUM(query_val * emb_val) / (
                SQRT((SELECT SUM(query_val * query_val) FROM UNNEST(query_vec) query_val)) *
                SQRT((SELECT SUM(emb_val * emb_val) FROM UNNEST(embedding) emb_val))
              )
              FROM UNNEST(query_vec) query_val WITH OFFSET pos1
              JOIN UNNEST(embedding) emb_val WITH OFFSET pos2
              ON pos1 = pos2
            ) AS similarity
          FROM `{self.full_table_id}`, query_embedding
          WHERE TRUE {source_filter}
        )
        SELECT
          id,
          source_type,
          source_name,
          content,
          similarity,
          metadata
        FROM similarities
        WHERE similarity >= {min_similarity}
        ORDER BY similarity DESC
        LIMIT {top_k}
        """

        try:
            query_job = self.client.query(query)
            results = query_job.result()

            similar_items = []
            for row in results:
                # BigQuery JSON type is already parsed as dict
                metadata = row["metadata"] if row["metadata"] else {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                similar_items.append({
                    "id": row["id"],
                    "source_type": row["source_type"],
                    "source_name": row["source_name"],
                    "content": row["content"],
                    "similarity": float(row["similarity"]) if row["similarity"] else 0.0,
                    "metadata": metadata
                })

            return similar_items

        except Exception as e:
            print(f"❌ Error searching embeddings: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_by_type(self, source_type: str) -> List[Dict]:
        """
        Get all embeddings of a specific type.

        Args:
            source_type: 'table' or 'pdf'

        Returns:
            List of all embeddings for that type
        """
        query = f"""
        SELECT id, source_name, content, metadata
        FROM `{self.full_table_id}`
        WHERE source_type = '{source_type}'
        ORDER BY created_at DESC
        """

        try:
            results = self.client.query(query).result()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"❌ Error fetching embeddings: {e}")
            return []

    def delete_by_type(self, source_type: str) -> bool:
        """Delete all embeddings of a specific type."""
        query = f"""
        DELETE FROM `{self.full_table_id}`
        WHERE source_type = '{source_type}'
        """

        try:
            self.client.query(query).result()
            print(f"✓ Deleted all {source_type} embeddings")
            return True
        except Exception as e:
            print(f"❌ Error deleting embeddings: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get statistics about stored embeddings."""
        query = f"""
        SELECT
          source_type,
          COUNT(*) as count,
          MIN(created_at) as oldest,
          MAX(created_at) as newest
        FROM `{self.full_table_id}`
        GROUP BY source_type
        """

        try:
            results = self.client.query(query).result()
            stats = {}
            for row in results:
                stats[row["source_type"]] = {
                    "count": row["count"],
                    "oldest": row["oldest"],
                    "newest": row["newest"]
                }
            return stats
        except Exception as e:
            print(f"❌ Error fetching stats: {e}")
            return {}
