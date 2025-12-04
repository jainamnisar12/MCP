"""
Query Text Embeddings - Search for similar text chunks in BigQuery
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings


def search_similar_text(query: str, top_k: int = 5, source_type: str = "text"):
    """
    Search for text chunks similar to the query.
    
    Args:
        query: The search query text
        top_k: Number of results to return
        source_type: Filter by source type (default: "text")
    """
    print(f"\n{'='*70}")
    print(f"🔍 SEARCHING TEXT EMBEDDINGS")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    print(f"{'='*70}\n")
    
    # Initialize embeddings model
    print("Initializing embeddings model...")
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION
    )
    
    # Generate query embedding
    print("Generating query embedding...")
    query_embedding = embeddings.embed_query(query)
    print(f"✓ Query embedding generated (dimension: {len(query_embedding)})\n")
    
    # Initialize BigQuery client
    client = bigquery.Client(project=config.GCP_PROJECT_ID)
    table_id = f"{config.GCP_PROJECT_ID}.{config.BIGQUERY_DATASET}.vector_embeddings_new"
    
    # Build SQL query with vector similarity
    sql = f"""
    WITH query_embedding AS (
        SELECT {query_embedding} AS embedding
    )
    SELECT 
        id,
        source_type,
        source_name,
        content,
        metadata,
        created_at,
        -- Calculate cosine similarity
        (
            SELECT SUM(e1 * e2) / (
                SQRT(SUM(e1 * e1)) * SQRT(SUM(e2 * e2))
            )
            FROM UNNEST(t.embedding) AS e1 WITH OFFSET pos1
            JOIN UNNEST(q.embedding) AS e2 WITH OFFSET pos2
            ON pos1 = pos2
        ) AS similarity_score
    FROM `{table_id}` t
    CROSS JOIN query_embedding q
    WHERE source_type = '{source_type}'
    ORDER BY similarity_score DESC
    LIMIT {top_k}
    """
    
    print("Executing search query...")
    query_job = client.query(sql)
    results = list(query_job.result())
    
    if not results:
        print("\n❌ No results found")
        return []
    
    print(f"\n✓ Found {len(results)} results\n")
    print("="*70)
    print("SEARCH RESULTS")
    print("="*70)
    
    for idx, row in enumerate(results, 1):
        print(f"\n[Result {idx}]")
        print(f"Similarity Score: {row.similarity_score:.4f}")
        print(f"Source: {row.source_name}")
        print(f"ID: {row.id}")
        print(f"Created: {row.created_at}")
        print(f"\nContent Preview:")
        content_preview = row.content[:300] + "..." if len(row.content) > 300 else row.content
        print(content_preview)
        print("-" * 70)
    
    return results


def list_all_text_embeddings():
    """List all text embeddings in the table."""
    print(f"\n{'='*70}")
    print(f"📋 LISTING ALL TEXT EMBEDDINGS")
    print(f"{'='*70}\n")
    
    client = bigquery.Client(project=config.GCP_PROJECT_ID)
    table_id = f"{config.GCP_PROJECT_ID}.{config.BIGQUERY_DATASET}.vector_embeddings_new"
    
    sql = f"""
    SELECT 
        id,
        source_type,
        source_name,
        LEFT(content, 100) AS content_preview,
        ARRAY_LENGTH(embedding) AS embedding_dimension,
        created_at
    FROM `{table_id}`
    WHERE source_type = 'text'
    ORDER BY created_at DESC
    """
    
    query_job = client.query(sql)
    results = list(query_job.result())
    
    if not results:
        print("❌ No text embeddings found in the table")
        return
    
    print(f"✓ Found {len(results)} text embeddings\n")
    
    for idx, row in enumerate(results, 1):
        print(f"[{idx}] {row.id}")
        print(f"    Source: {row.source_name}")
        print(f"    Dimension: {row.embedding_dimension}")
        print(f"    Created: {row.created_at}")
        print(f"    Preview: {row.content_preview}...")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query text embeddings in BigQuery")
    parser.add_argument("--list", action="store_true", help="List all text embeddings")
    parser.add_argument("--search", type=str, help="Search query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.list:
        list_all_text_embeddings()
    elif args.search:
        search_similar_text(args.search, args.top_k)
    else:
        # Default: show example searches
        print("\n" + "="*70)
        print("TEXT EMBEDDINGS QUERY TOOL")
        print("="*70)
        print("\nUsage:")
        print("  python query_text_embeddings.py --list")
        print("  python query_text_embeddings.py --search 'your query' --top-k 5")
        print("\nExamples:")
        print("  python query_text_embeddings.py --search 'UPI payment methods'")
        print("  python query_text_embeddings.py --search 'one click checkout' --top-k 3")
        print("\n" + "="*70)
        
        # Run example search
        print("\n\n🔍 Running example search...\n")
        search_similar_text("UPI one-click payment", top_k=3)
