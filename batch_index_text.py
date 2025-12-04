"""
Batch Text Embeddings - Add multiple text chunks to BigQuery vector store
"""

from index_text_chunks import TextEmbeddingIndexer

# ============================================================================
# ADD YOUR TEXT CHUNKS HERE
# ============================================================================

# Example: UPI Payment Related Text Chunks
text_chunks_to_index = [
    # Chunk 1: XPress UPI Payment Guide (already indexed above)
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
Next Action: Tap "UPI 1-CLICK PAYMENT" to proceed to authentication and finalize your order.""",
    
    # Add more chunks below:
    # """Your second text chunk here...""",
    # """Your third text chunk here...""",
]

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source identifier (used to track where these chunks came from)
SOURCE_NAME = "upi_payment_documentation"

# Optional metadata to attach to all chunks
METADATA = {
    "category": "upi_payment",
    "language": "english",
    "created_by": "manual_input"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📝 BATCH TEXT EMBEDDINGS INDEXER")
    print("="*70)
    print(f"Total chunks to process: {len(text_chunks_to_index)}")
    print(f"Source name: {SOURCE_NAME}")
    print("="*70 + "\n")
    
    # Initialize indexer with text-embedding-004 model
    indexer = TextEmbeddingIndexer(model_name="text-embedding-004")
    
    # Index all text chunks
    success = indexer.index_text_chunks(
        text_chunks=text_chunks_to_index,
        source_name=SOURCE_NAME,
        metadata=METADATA
    )
    
    if success:
        print("\n✅ All text chunks successfully indexed to BigQuery!")
        print(f"   Table: vector_embeddings_new")
        print(f"   Model: text-embedding-004")
        print(f"   Dimension: 768")
    else:
        print("\n❌ Failed to index some or all text chunks")
