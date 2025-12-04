"""
Text Chunk Embeddings Indexer - Creates embeddings for text chunks and stores in BigQuery
Uses Google's text-embedding-004 model for generating embeddings
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from index_new_pdf import NewBigQueryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from datetime import datetime
import hashlib
from typing import List, Dict


class TextEmbeddingIndexer:
    """Creates embeddings for text chunks and stores them in BigQuery."""

    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initialize the text embedding indexer.
        
        Args:
            model_name: The embedding model to use (default: text-embedding-004)
        """
        print(f"\n{'='*70}")
        print(f"📝  TEXT CHUNK EMBEDDINGS INDEXER")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Project: {config.GCP_PROJECT_ID}")
        print(f"Dataset: {config.BIGQUERY_DATASET}")
        print(f"Table: vector_embeddings_new")
        print(f"{'='*70}\n")

        # Initialize embeddings model
        self.embeddings = VertexAIEmbeddings(
            model_name=model_name,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION
        )
        
        # Initialize BigQuery vector store
        self.vector_store = NewBigQueryVectorStore()
        
        print("✓ Initialized embeddings model and vector store\n")

    def generate_chunk_id(self, content: str, source_name: str) -> str:
        """
        Generate a unique ID for a text chunk.
        
        Args:
            content: The text content
            source_name: The source name/identifier
            
        Returns:
            A unique hash-based ID
        """
        content_hash = hashlib.md5(f"{source_name}_{content}".encode()).hexdigest()
        return f"text_chunk_{content_hash[:12]}"

    def index_text_chunks(
        self,
        text_chunks: List[str],
        source_name: str = "manual_input",
        metadata: Dict = None
    ) -> bool:
        """
        Generate embeddings for text chunks and store in BigQuery.
        
        Args:
            text_chunks: List of text strings to embed
            source_name: Name/identifier for the source of these chunks
            metadata: Optional metadata dictionary to store with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not text_chunks:
            print("❌ No text chunks provided")
            return False

        print(f"[1/4] Processing {len(text_chunks)} text chunks...")
        print(f"      Source: {source_name}\n")

        # Ensure table exists
        print("[2/4] Ensuring BigQuery table exists...")
        self.vector_store.create_embeddings_table()
        print()

        # Generate embeddings
        print("[3/4] Generating embeddings...")
        embeddings_data = []
        
        for idx, chunk in enumerate(text_chunks, 1):
            try:
                # Generate embedding
                print(f"  Processing chunk {idx}/{len(text_chunks)}...")
                print(f"    Content preview: {chunk[:100]}...")
                
                embedding_vector = self.embeddings.embed_query(chunk)
                
                # Create embedding data
                chunk_id = self.generate_chunk_id(chunk, source_name)
                chunk_data = {
                    "id": chunk_id,
                    "source_name": source_name,
                    "content": chunk,
                    "embedding": embedding_vector,
                    "metadata": {
                        "chunk_index": idx,
                        "chunk_length": len(chunk),
                        "model": "text-embedding-004",
                        "timestamp": datetime.utcnow().isoformat(),
                        **(metadata or {})
                    }
                }
                embeddings_data.append(chunk_data)
                
                print(f"    ✓ Generated embedding (dimension: {len(embedding_vector)})")
                
            except Exception as e:
                print(f"    ❌ Error processing chunk {idx}: {e}")
                continue

        if not embeddings_data:
            print("\n❌ No embeddings generated")
            return False

        print(f"\n✓ Generated {len(embeddings_data)} embeddings")
        print()

        # Store in BigQuery
        print("[4/4] Storing embeddings in BigQuery...")
        success = self.vector_store.insert_embeddings(
            embeddings_data=embeddings_data,
            source_type="text"
        )

        if success:
            print(f"\n{'='*70}")
            print(f"✅ SUCCESS! Indexed {len(embeddings_data)} text chunks to BigQuery")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"❌ FAILED to index text chunks")
            print(f"{'='*70}\n")

        return success


def main():
    """Main execution function."""
    
    # Sample text chunks - You can replace this with your own data
    text_chunks = [
        """Xpress UPI Payment - 4 Step Payment Method Selection
Key Features of Express UPI Payment:
* Preferred Payment Option: The screen displays XPressUPI as the primary payment method, positioned at the top for quick access
* Pre-linked Bank Account: Your bank account (ending in 8812) is already connected to the UPI system, eliminating the need for manual entry
* One-Click Payment Button: The prominent "UPI 1-CLICK PAYMENT" button enables instant payment completion without redirecting to external apps
* Shipping Details Confirmed: Recipient (Manoj B.) and delivery address (Lal Bahadur Shastri Marg, Kanjurmarg West, Mumbai) are displayed for verification
* Secure Transaction: UPI integration ensures bank-level security with encrypted transactions
* No App Switching Required: Unlike traditional UPI payments that open external apps, this express method processes payment within the same interface
* Time-Saving: Reduces payment time from 30-40 seconds to under 5 seconds
* Quick Checkout Flow: Simply tap the blue button to authenticate and complete payment instantly
Next Action: Tap "UPI 1-CLICK PAYMENT" to proceed to authentication and finalize your order."""
    ]
    
    # Initialize indexer
    indexer = TextEmbeddingIndexer(model_name="text-embedding-004")
    
    # Index the text chunks
    success = indexer.index_text_chunks(
        text_chunks=text_chunks,
        source_name="xpress_upi_payment_guide",
        metadata={
            "category": "upi_payment",
            "feature": "xpress_payment",
            "document_type": "user_guide"
        }
    )
    
    if success:
        print("🎉 Text chunks successfully indexed!")
    else:
        print("⚠️  Indexing completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
