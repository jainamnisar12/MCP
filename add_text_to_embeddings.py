"""
Add Text Chunks to BigQuery Vector Embeddings
Similar pattern to index_websites.py but for manual text input
"""

import os
import sys
from datetime import datetime
from typing import List, Dict
import hashlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from langchain_google_vertexai import VertexAIEmbeddings
from index_new_pdf import NewBigQueryVectorStore


def generate_unique_id(text: str, source_name: str) -> str:
    """Generate unique ID from text content."""
    content_hash = hashlib.md5(f"{source_name}:{text}".encode()).hexdigest()
    return f"text_{content_hash[:16]}"


def add_text_chunks_to_bigquery(
    text_chunks: List[str],
    source_name: str,
    metadata: Dict = None
) -> bool:
    """
    Add text chunks to BigQuery vector embeddings table.
    
    Args:
        text_chunks: List of text strings to embed
        source_name: Identifier for the source (e.g., "xpress_upi_guide")
        metadata: Optional additional metadata
        
    Returns:
        True if successful
    """
    print("\n" + "="*80)
    print("📝 ADDING TEXT CHUNKS TO BIGQUERY VECTOR EMBEDDINGS")
    print("="*80)
    print(f"Source: {source_name}")
    print(f"Number of chunks: {len(text_chunks)}")
    print(f"Model: gemini-embedding-001")
    print(f"Target table: vector_embeddings_new")
    print("="*80 + "\n")
    
    # Initialize components
    print("[1/4] Initializing embeddings and BigQuery client...")
    try:
        embeddings = VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION
        )
        vector_store = NewBigQueryVectorStore()
        print("  ✓ Components initialized\n")
    except Exception as e:
        print(f"  ❌ Error initializing: {e}\n")
        return False
    
    # Ensure table exists
    print("[2/4] Ensuring BigQuery table exists...")
    vector_store.create_embeddings_table()
    print()
    
    # Generate embeddings
    print("[3/4] Generating embeddings for text chunks...")
    embeddings_data = []
    
    for idx, text_chunk in enumerate(text_chunks, 1):
        try:
            print(f"\n  Processing chunk {idx}/{len(text_chunks)}:")
            preview = text_chunk[:100].replace('\n', ' ')
            print(f"    Preview: {preview}...")
            
            # Generate embedding
            embedding_vector = embeddings.embed_query(text_chunk)
            
            # Create unique ID
            chunk_id = generate_unique_id(text_chunk, source_name)
            
            # Prepare metadata
            chunk_metadata = {
                "chunk_index": idx,
                "chunk_length": len(text_chunk),
                "model": "gemini-embedding-001",
                "embedding_dimension": len(embedding_vector),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add custom metadata if provided
            if metadata:
                chunk_metadata.update(metadata)
            
            # Create embedding data entry
            embedding_entry = {
                "id": chunk_id,
                "source_name": source_name,
                "content": text_chunk,
                "embedding": embedding_vector,
                "metadata": chunk_metadata
            }
            
            embeddings_data.append(embedding_entry)
            print(f"    ✓ Embedding generated (dimension: {len(embedding_vector)})")
            
        except Exception as e:
            print(f"    ❌ Error processing chunk {idx}: {e}")
            continue
    
    if not embeddings_data:
        print("\n❌ No embeddings generated successfully")
        return False
    
    print(f"\n  ✓ Successfully generated {len(embeddings_data)} embeddings\n")
    
    # Insert into BigQuery
    print("[4/4] Inserting embeddings into BigQuery...")
    try:
        success = vector_store.insert_embeddings(
            embeddings_data=embeddings_data,
            source_type="website"  # Changed from "text" to "website" so agent picks it up
        )
        
        if success:
            print("\n" + "="*80)
            print("✅ SUCCESS!")
            print("="*80)
            print(f"  Inserted {len(embeddings_data)} text chunks into BigQuery")
            print(f"  Table: {config.BIGQUERY_DATASET}.vector_embeddings_new")
            print(f"  Source: {source_name}")
            print("="*80 + "\n")
            return True
        else:
            print("\n❌ Failed to insert embeddings into BigQuery\n")
            return False
            
    except Exception as e:
        print(f"\n❌ Error inserting into BigQuery: {e}\n")
        return False


def main():
    """Main execution function."""
    
    # Define your text chunks here
    text_chunks = [
        """
        Page: Steps for XPress UPI Payment method selection.
        Xpress UPI Payment - 4 Step Payment Method Selection
Key Features of Express UPI Payment:
* Preferred Payment Option: The screen displays XPressUPI as the primary payment method, positioned at the top for quick access
* Pre-linked Bank Account: Your bank account (ending in 8812) is already connected to the UPI system, eliminating the need for manual entry
* One-Click Payment Button: The prominent "UPI 1-CLICK PAYMENT" button enables instant payment completion without redirecting to external apps
* Shipping Details Confirmed: Recipient (Manoj B.) and delivery address (Lal Bahadur Shastri Marg, Kanjurmarg West, Mumbai) are displayed for verification
* Secure Transaction: UPI integration ensures bank-level security with encrypted transactions
* No App Switching Required: Unlike traditional UPI payments that open external apps, this express method processes payment within the same interface
* Time-Saving: Reduces payment time from 30-40 seconds to under 5 seconds
* Quick Checkout Flow: Simply tap the blue button to authenticate and complete payment instantly
Next Action: Tap "UPI 1-CLICK PAYMENT" to proceed to authentication and finalize your order.""",
        
        # Add more text chunks here as needed:
        # """Your second text chunk here...""",
        # """Your third text chunk here...""",
    ]
    
    # Configuration
    source_name = "xpress_upi_payment_guide"
    metadata = {
        "category": "upi_payment",
        "feature": "xpress_payment",
        "document_type": "user_guide",
        "language": "english"
    }
    
    # Process the chunks
    success = add_text_chunks_to_bigquery(
        text_chunks=text_chunks,
        source_name=source_name,
        metadata=metadata
    )
    
    if success:
        print("🎉 Text chunks successfully added to BigQuery!")
    else:
        print("⚠️  Failed to add text chunks to BigQuery")
        sys.exit(1)


if __name__ == "__main__":
    main()
