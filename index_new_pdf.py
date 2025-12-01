"""
PDF Indexer for New UPI Document - Creates embeddings and stores in new BigQuery table
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from google.cloud import bigquery
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from datetime import datetime
import json


class NewBigQueryVectorStore:
    """Manages vector embeddings in the new BigQuery table."""

    def __init__(self, dataset_name: str = None, table_name: str = "vector_embeddings_new"):
        self.client = bigquery.Client(project=config.GCP_PROJECT_ID)
        self.dataset_name = dataset_name or config.BIGQUERY_DATASET
        self.table_name = table_name
        self.full_table_id = f"{config.GCP_PROJECT_ID}.{self.dataset_name}.{self.table_name}"

    def create_embeddings_table(self):
        """Create the new embeddings table."""
        
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
        table.description = "Vector embeddings for UPI Transaction Process Explained PDF"

        try:
            table = self.client.create_table(table)
            print(f"✓ Created table {self.full_table_id}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"✓ Table {self.full_table_id} already exists")
                return True
            else:
                print(f"❌ Error creating table: {e}")
                return False

    def insert_embeddings(self, embeddings_data: list, source_type: str = "pdf"):
        """Insert embeddings into BigQuery table."""
        
        if not embeddings_data:
            print("❌ No embeddings data to insert")
            return False

        # Prepare rows for insertion
        rows_to_insert = []
        for data in embeddings_data:
            row = {
                "id": data["id"],
                "source_type": source_type,
                "source_name": data["source_name"],
                "content": data["content"],
                "embedding": data["embedding"],
                "metadata": json.dumps(data.get("metadata", {})),
                "created_at": datetime.utcnow().isoformat()
            }
            rows_to_insert.append(row)

        # Insert in batches
        batch_size = 100
        total_batches = (len(rows_to_insert) + batch_size - 1) // batch_size

        print(f"Inserting {len(rows_to_insert)} rows in {total_batches} batches...")

        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                errors = self.client.insert_rows_json(
                    table=self.client.get_table(self.full_table_id),
                    json_rows=batch
                )

                if errors:
                    print(f"❌ Batch {batch_num}/{total_batches} errors: {errors}")
                    return False
                else:
                    print(f"✓ Batch {batch_num}/{total_batches} inserted successfully")

            except Exception as e:
                print(f"❌ Error inserting batch {batch_num}: {e}")
                return False

        print(f"✅ All {len(rows_to_insert)} embeddings inserted successfully!")
        return True

    def similarity_search(self, query_embedding: list, k: int = 5, similarity_threshold: float = 0.7, source_type: str = "pdf"):
        """Perform similarity search using cosine similarity."""
        
        print(f"🔍 BigQuery similarity search starting...")
        print(f"   • Query embedding dimensions: {len(query_embedding)}")
        print(f"   • Query embedding (first 5): {query_embedding[:5]}")
        print(f"   • Similarity threshold: {similarity_threshold}")
        print(f"   • Max results (k): {k}")
        print(f"   • Table: {self.full_table_id}")
        
        # Convert query embedding to string format for SQL
        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

        query = f"""
        WITH similarity_scores AS (
            SELECT 
                id,
                source_name,
                content,
                metadata,
                embedding,
                (
                    SELECT SUM(a * b) / (
                        SQRT(SUM(a * a)) * SQRT(SUM(b * b))
                    )
                    FROM (
                        SELECT 
                            embedding[OFFSET(i)] as a,
                            {query_vector_str}[OFFSET(i)] as b
                        FROM UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(embedding) - 1)) as i
                    )
                ) as similarity_score
            FROM `{self.full_table_id}`
            WHERE source_type = '{source_type}'
        )
        SELECT 
            id,
            source_name,
            content,
            metadata,
            similarity_score
        FROM similarity_scores
        WHERE similarity_score >= {similarity_threshold}
        ORDER BY similarity_score DESC
        LIMIT {k}
        """

        print(f"🔍 Executing BigQuery similarity search...")
        try:
            results = self.client.query(query).result()
            result_list = [
                {
                    "id": row.id,
                    "source_name": row.source_name,
                    "content": row.content,
                    "metadata": row.metadata if row.metadata else {},
                    "similarity_score": row.similarity_score
                }
                for row in results
            ]
            
            print(f"✅ BigQuery search completed")
            print(f"   • Found {len(result_list)} results")
            for i, result in enumerate(result_list):
                print(f"   • Result {i+1}: {result['source_name']} (score: {result['similarity_score']:.4f})")
            
            return result_list
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return []


def index_upi_pdf():
    """Index the UPI Transaction Process Explained PDF into new BigQuery table."""
    
    # Path to your UPI PDF (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "data", "UPI Transaction Process Explained.pdf")
    
    print("\n" + "="*60)
    print("📄 INDEXING UPI PDF TO NEW BIGQUERY TABLE")
    print("="*60)

    # Step 1: Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF not found at {pdf_path}")
        return False

    # Step 2: Load and split PDF
    print(f"\n[1/5] Loading PDF from {pdf_path}...")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} pages")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return False

    print("\n[2/5] Splitting PDF into chunks...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)
        print(f"✓ Created {len(docs)} text chunks")
    except Exception as e:
        print(f"❌ Error splitting PDF: {e}")
        return False

    # Step 3: Initialize Vertex AI embeddings
    print("\n[3/5] Initializing Vertex AI embeddings...")
    try:
        embeddings_model = VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION if hasattr(config, 'GCP_LOCATION') else "us-central1"
        )
        print("✓ Vertex AI embeddings initialized")
    except Exception as e:
        print(f"❌ Error initializing Vertex AI embeddings: {e}")
        return False

    # Step 4: Create embeddings data
    print("\n[4/5] Creating embeddings for PDF chunks...")
    embeddings_data = []
    pdf_name = os.path.basename(pdf_path)

    for idx, doc in enumerate(docs):
        try:
            embedding = embeddings_model.embed_query(doc.page_content)

            embeddings_data.append({
                "id": f"upi_pdf_chunk_{idx}",
                "source_name": f"{pdf_name}_chunk_{idx}",
                "content": doc.page_content,
                "embedding": embedding,
                "metadata": {
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": idx,
                    "pdf_name": pdf_name,
                    "source": doc.metadata.get("source", ""),
                    "file_path": pdf_path
                }
            })

            if (idx + 1) % 10 == 0:
                print(f"  ✓ Created {idx + 1}/{len(docs)} embeddings...")

        except Exception as e:
            print(f"  ❌ Error creating embedding for chunk {idx}: {e}")
            continue

    print(f"✓ Created {len(embeddings_data)} PDF chunk embeddings")

    # Step 5: Log chunks instead of storing in BigQuery
    print("\n[5/5] Logging chunks (BigQuery insertion commented out)...")
    print("\n" + "="*80)
    print("📦 GENERATED PDF CHUNKS:")
    print("="*80)
    
    for i, chunk_data in enumerate(embeddings_data):
        print(f"\n{'─'*80}")
        print(f"CHUNK {i+1}/{len(embeddings_data)}")
        print(f"{'─'*80}")
        print(f"ID: {chunk_data['id']}")
        print(f"Source: {chunk_data['source_name']}")
        print(f"Page: {chunk_data['metadata'].get('page', 'N/A')}")
        print(f"Chunk Index: {chunk_data['metadata']['chunk_index']}")
        print(f"Content Length: {len(chunk_data['content'])} chars")
        print(f"Embedding Dimensions: {len(chunk_data['embedding'])}")
        print(f"\nCONTENT:")
        print(f"{'-'*80}")
        print(chunk_data['content'])
        print(f"{'-'*80}")
    
    print(f"\n{'='*80}\n")

    # COMMENTED OUT: BigQuery insertion
    # print("\n[5/5] Storing embeddings in new BigQuery table...")
    # vector_store = NewBigQueryVectorStore(dataset_name=config.BIGQUERY_DATASET)
    # 
    # # Create table
    # vector_store.create_embeddings_table()
    # 
    # # Insert embeddings
    # success = vector_store.insert_embeddings(
    #     embeddings_data=embeddings_data,
    #     source_type="pdf"
    # )
    # 
    # if not success:
    #     print("❌ Failed to insert embeddings")
    #     return False

    print("\n" + "="*60)
    print("✅ PDF CHUNKS LOGGED SUCCESSFULLY (NOT PUSHED TO BIGQUERY)")
    print("="*60)
    print(f"📄 PDF: {os.path.basename(pdf_path)}")
    print(f"📊 Total chunks logged: {len(embeddings_data)}")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = index_upi_pdf()
    sys.exit(0 if success else 1)
