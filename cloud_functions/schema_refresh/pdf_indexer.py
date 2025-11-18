"""
PDF indexer module for Cloud Function
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from bigquery_vector_store import BigQueryVectorStore


def create_pdf_embeddings_internal(
    project_id: str,
    dataset: str,
    location: str = "us-central1",
    pdf_path: str = "/workspace/data/UPI Transaction Process Explained.pdf"
) -> dict:
    """
    Create and store PDF embeddings in BigQuery.

    This is an internal function meant to be called by Cloud Functions.

    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        location: GCP location
        pdf_path: Path to PDF file (default: /workspace/data/...)

    Returns:
        Dict with success status and details
    """
    try:
        # Step 1: Load and split PDF
        print(f"  Loading PDF from {pdf_path}...")

        # Check if PDF exists
        if not os.path.exists(pdf_path):
            print(f"  ⚠️  PDF not found at {pdf_path}, skipping PDF indexing")
            return {
                'success': True,
                'chunks_indexed': 0,
                'skipped': True,
                'message': 'PDF file not found'
            }

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"  ✓ Loaded {len(documents)} pages")

        # Step 2: Split into chunks
        print("  Splitting PDF into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)
        print(f"  ✓ Created {len(docs)} chunks")

        # Step 3: Create embeddings
        print("  Initializing Vertex AI embeddings...")
        embeddings_model = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id,
            location=location
        )

        print("  Creating embeddings for PDF chunks...")
        embeddings_data = []
        pdf_name = os.path.basename(pdf_path)

        for idx, doc in enumerate(docs):
            embedding = embeddings_model.embed_query(doc.page_content)

            # Prepare metadata - ensure all values are JSON-compatible
            metadata = {
                "page": int(doc.metadata.get("page", 0)),
                "chunk_index": int(idx),
                "pdf_name": str(pdf_name),
                "source": str(doc.metadata.get("source", ""))
            }

            embeddings_data.append({
                "id": f"pdf_chunk_{idx}",
                "source_name": f"{pdf_name}_chunk_{idx}",
                "content": doc.page_content,
                "embedding": embedding,
                "metadata": metadata
            })

            if (idx + 1) % 10 == 0:
                print(f"    ✓ Created {idx + 1}/{len(docs)} embeddings...")

        print(f"  ✓ Created {len(embeddings_data)} PDF chunk embeddings")

        # Step 4: Initialize BigQuery Vector Store and insert
        print("  Initializing BigQuery Vector Store...")
        vector_store = BigQueryVectorStore(dataset_name=dataset)
        vector_store.create_embeddings_table()

        print("  Inserting embeddings into BigQuery...")
        success = vector_store.insert_embeddings(
            embeddings_data=embeddings_data,
            source_type="pdf"
        )

        if success:
            return {
                'success': True,
                'chunks_indexed': len(embeddings_data),
                'embedding_dimensions': len(embeddings_data[0]['embedding']) if embeddings_data else 0
            }
        else:
            return {
                'success': False,
                'message': 'Failed to insert embeddings'
            }

    except Exception as e:
        print(f"  ❌ Error in PDF indexing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }
