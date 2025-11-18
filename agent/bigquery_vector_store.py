"""
BigQuery Vector Store for embeddings storage and retrieval
Supports both table and PDF embeddings in a unified table
"""

import os
import json
from typing import List, Dict, Optional
from google.cloud import bigquery
from datetime import datetime


class BigQueryVectorStore:
    """
    Unified BigQuery storage for both PDF and table embeddings.
    Uses a single table with source_type field to differentiate between types.
    """

    def __init__(
        self,
        dataset_name: str = None,
        table_name: str = "vector_embeddings",
        project_id: str = None
    ):
        """
        Initialize BigQuery Vector Store.

        Args:
            dataset_name: BigQuery dataset name
            table_name: Table name for storing embeddings
            project_id: GCP project ID (optional, uses default if not provided)
        """
        self.project_id = project_id or os.environ.get('GCP_PROJECT_ID')
        self.dataset_name = dataset_name or os.environ.get('BIGQUERY_DATASET')
        self.table_name = table_name

        if not self.project_id or not self.dataset_name:
            raise ValueError("project_id and dataset_name are required")

        self.client = bigquery.Client(project=self.project_id)
        self.table_id = f"{self.project_id}.{self.dataset_name}.{self.table_name}"

    def create_embeddings_table(self):
        """
        Create the embeddings table if it doesn't exist.

        Schema:
            - id: Unique identifier (e.g., "table_customers", "pdf_chunk_0")
            - source_type: Type of source ("table" or "pdf")
            - source_name: Name of the source (table name or chunk ID)
            - content: The text content that was embedded
            - embedding: The embedding vector (ARRAY<FLOAT64>)
            - metadata: Additional metadata (JSON)
            - created_at: Timestamp when created
        """
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]

        table = bigquery.Table(self.table_id, schema=schema)

        try:
            self.client.get_table(self.table_id)
            print(f"  ✓ Table {self.table_id} already exists")
        except Exception:
            table = self.client.create_table(table)
            print(f"  ✓ Created table {self.table_id}")

    def insert_embeddings(
        self,
        embeddings_data: List[Dict],
        source_type: str
    ) -> bool:
        """
        Insert embeddings into BigQuery.

        First deletes existing embeddings of the same source_type,
        then inserts new embeddings.

        Args:
            embeddings_data: List of dicts with keys: id, source_name, content, embedding, metadata
            source_type: Type of source ("table" or "pdf")

        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Delete existing embeddings of this source_type
            delete_query = f"""
                DELETE FROM `{self.table_id}`
                WHERE source_type = @source_type
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("source_type", "STRING", source_type)
                ]
            )

            delete_job = self.client.query(delete_query, job_config=job_config)
            delete_job.result()
            print(f"  ✓ Deleted existing {source_type} embeddings")

            # Step 2: Insert rows using parameterized query (better JSON support)
            current_time = datetime.utcnow()

            # Insert each row using a parameterized query
            for idx, item in enumerate(embeddings_data):
                insert_query = f"""
                    INSERT INTO `{self.table_id}`
                    (id, source_type, source_name, content, embedding, metadata, created_at)
                    VALUES (@id, @source_type, @source_name, @content, @embedding, PARSE_JSON(@metadata), @created_at)
                """

                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("id", "STRING", item["id"]),
                        bigquery.ScalarQueryParameter("source_type", "STRING", source_type),
                        bigquery.ScalarQueryParameter("source_name", "STRING", item["source_name"]),
                        bigquery.ScalarQueryParameter("content", "STRING", item["content"]),
                        bigquery.ArrayQueryParameter("embedding", "FLOAT64", item["embedding"]),
                        bigquery.ScalarQueryParameter("metadata", "STRING", json.dumps(item.get("metadata", {}))),
                        bigquery.ScalarQueryParameter("created_at", "TIMESTAMP", current_time)
                    ]
                )

                try:
                    query_job = self.client.query(insert_query, job_config=job_config)
                    query_job.result()

                    if (idx + 1) % 10 == 0:
                        print(f"    ✓ Inserted {idx + 1}/{len(embeddings_data)} embeddings...")
                except Exception as e:
                    print(f"  ❌ Error inserting row {idx}: {e}")
                    return False

            print(f"  ✓ Inserted {len(embeddings_data)} {source_type} embeddings")
            return True

        except Exception as e:
            print(f"  ❌ Error inserting embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search_similar(
        self,
        query_embedding: List[float],
        source_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar embeddings using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            source_type: Filter by source type ("table", "pdf", or None for all)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of similar items with their similarity scores
        """
        # Convert embedding to SQL array format
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Build query with optional source_type filter
        where_clause = ""
        if source_type:
            where_clause = f"WHERE source_type = '{source_type}'"

        query = f"""
        WITH query_embedding AS (
            SELECT {embedding_str} AS embedding
        ),
        similarities AS (
            SELECT
                id,
                source_type,
                source_name,
                content,
                metadata,
                (
                    SELECT SUM(a * b)
                    FROM UNNEST(e.embedding) AS a WITH OFFSET pos1
                    JOIN UNNEST(q.embedding) AS b WITH OFFSET pos2
                    ON pos1 = pos2
                ) / (
                    SQRT((SELECT SUM(a * a) FROM UNNEST(e.embedding) AS a)) *
                    SQRT((SELECT SUM(b * b) FROM UNNEST(q.embedding) AS b))
                ) AS similarity
            FROM `{self.table_id}` e
            CROSS JOIN query_embedding q
            {where_clause}
        )
        SELECT *
        FROM similarities
        WHERE similarity >= {min_similarity}
        ORDER BY similarity DESC
        LIMIT {top_k}
        """

        try:
            query_job = self.client.query(query)
            results = []

            for row in query_job:
                # Handle metadata - BigQuery returns it as native dict
                metadata = row["metadata"] if row["metadata"] else {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                results.append({
                    "id": row["id"],
                    "source_type": row["source_type"],
                    "source_name": row["source_name"],
                    "content": row["content"],
                    "metadata": metadata,
                    "similarity": float(row["similarity"])
                })

            return results

        except Exception as e:
            print(f"  ❌ Error searching embeddings: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_stats(self) -> Dict:
        """
        Get statistics about stored embeddings.

        Returns:
            Dict with counts by source_type
        """
        query = f"""
        SELECT
            source_type,
            COUNT(*) as count
        FROM `{self.table_id}`
        GROUP BY source_type
        ORDER BY source_type
        """

        try:
            query_job = self.client.query(query)
            stats = {}

            for row in query_job:
                stats[row["source_type"]] = row["count"]

            return stats

        except Exception as e:
            print(f"  ❌ Error getting stats: {e}")
            return {}
