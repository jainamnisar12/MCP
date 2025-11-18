"""
PDF Indexer for BigQuery - Creates embeddings for PDF content and stores in BigQuery
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bigquery_vector_store import BigQueryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings


def create_pdf_embeddings():
    """Create embeddings for PDF content and store in BigQuery."""

    print("\n" + "="*60)
    print("📄 PDF INDEXING TO BIGQUERY")
    print("="*60)

    # Step 1: Check if PDF exists
    if not os.path.exists(config.PDF_PATH):
        print(f"❌ Error: PDF not found at {config.PDF_PATH}")
        return False

    # Step 2: Load and split PDF
    print(f"\n[1/5] Loading PDF from {config.PDF_PATH}...")
    try:
        loader = PyPDFLoader(config.PDF_PATH)
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
            model_name="text-embedding-004",
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
    pdf_name = os.path.basename(config.PDF_PATH)

    for idx, doc in enumerate(docs):
        try:
            embedding = embeddings_model.embed_query(doc.page_content)

            embeddings_data.append({
                "id": f"pdf_chunk_{idx}",
                "source_name": f"{pdf_name}_chunk_{idx}",
                "content": doc.page_content,
                "embedding": embedding,
                "metadata": {
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": idx,
                    "pdf_name": pdf_name,
                    "source": doc.metadata.get("source", "")
                }
            })

            if (idx + 1) % 10 == 0:
                print(f"  ✓ Created {idx + 1}/{len(docs)} embeddings...")

        except Exception as e:
            print(f"  ❌ Error creating embedding for chunk {idx}: {e}")
            continue

    print(f"✓ Created {len(embeddings_data)} PDF chunk embeddings")

    # Step 5: Initialize BigQuery Vector Store and insert
    print("\n[5/5] Storing embeddings in BigQuery...")
    try:
        vector_store = BigQueryVectorStore()
        vector_store.create_embeddings_table()

        success = vector_store.insert_embeddings(
            embeddings_data=embeddings_data,
            source_type="pdf"
        )

        if success:
            print("\n" + "="*60)
            print("✅ PDF EMBEDDINGS STORED IN BIGQUERY!")
            print("="*60)
            print(f"📄 PDF: {pdf_name}")
            print(f"📊 Total chunks indexed: {len(embeddings_data)}")
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
        print(f"❌ Error storing embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_pdf_embeddings()
    sys.exit(0 if success else 1)
