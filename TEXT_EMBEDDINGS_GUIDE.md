# Text Embeddings Indexer - Usage Guide

## Overview
This tool generates embeddings for text chunks using Google's **text-embedding-004** model and stores them in BigQuery's `vector_embeddings_new` table.

## Quick Start

### 1. Single Text Chunk
```bash
# Edit index_text_chunks.py and modify the text_chunks array
python index_text_chunks.py
```

### 2. Multiple Text Chunks (Batch)
```bash
# Edit batch_index_text.py and add your text chunks
python batch_index_text.py
```

## Files Created

### `index_text_chunks.py`
- **Purpose**: Core indexer class with example usage
- **Features**:
  - `TextEmbeddingIndexer` class for generating embeddings
  - Automatic table creation in BigQuery
  - Batch insertion with progress tracking
  - Error handling and logging

### `batch_index_text.py`
- **Purpose**: Easy-to-use batch processing script
- **Usage**: Simply add your text chunks to the array and run

## How to Add Text Chunks

### Option 1: Edit batch_index_text.py
```python
text_chunks_to_index = [
    """Your first text chunk here...""",
    """Your second text chunk here...""",
    """Your third text chunk here...""",
]
```

### Option 2: Programmatic Usage
```python
from index_text_chunks import TextEmbeddingIndexer

indexer = TextEmbeddingIndexer(model_name="text-embedding-004")

chunks = ["Text 1", "Text 2", "Text 3"]

indexer.index_text_chunks(
    text_chunks=chunks,
    source_name="my_source",
    metadata={"category": "documentation"}
)
```

## Configuration

### Model Settings
- **Model**: `text-embedding-004` (768 dimensions)
- **Provider**: Google Vertex AI
- **Project**: Configured via `config.GCP_PROJECT_ID`
- **Location**: Configured via `config.GCP_LOCATION`

### BigQuery Table
- **Table**: `vector_embeddings_new`
- **Schema**:
  - `id` (STRING): Unique identifier
  - `source_type` (STRING): Set to "text" for text chunks
  - `source_name` (STRING): Your custom source identifier
  - `content` (STRING): The actual text content
  - `embedding` (FLOAT64, REPEATED): 768-dimensional vector
  - `metadata` (JSON): Additional information
  - `created_at` (TIMESTAMP): Creation timestamp

## Features

### Automatic ID Generation
Each chunk gets a unique ID based on MD5 hash of source_name + content

### Metadata Support
Store additional information with each embedding:
```python
metadata = {
    "category": "upi_payment",
    "feature": "xpress_payment",
    "language": "english",
    "version": "1.0"
}
```

### Progress Tracking
Real-time feedback on:
- Chunk processing progress
- Embedding generation
- BigQuery insertion status

### Error Handling
- Continues processing even if individual chunks fail
- Detailed error messages for debugging
- Batch insertion with rollback protection

## Example Output

```
======================================================================
📝  TEXT CHUNK EMBEDDINGS INDEXER
======================================================================
Model: text-embedding-004
Project: ai-project-464806
Dataset: mindgatetestingdata2
Table: vector_embeddings_new
======================================================================

✓ Initialized embeddings model and vector store

[1/4] Processing 1 text chunks...
      Source: xpress_upi_payment_guide

[2/4] Ensuring BigQuery table exists...
✓ Table ai-project-464806.mindgatetestingdata2.vector_embeddings_new already exists

[3/4] Generating embeddings...
  Processing chunk 1/1...
    Content preview: Xpress UPI Payment - 4 Step Payment Method Selection...
    ✓ Generated embedding (dimension: 768)

✓ Generated 1 embeddings

[4/4] Storing embeddings in BigQuery...
Inserting 1 rows in 1 batches...
✓ Batch 1/1 inserted successfully

======================================================================
✅ SUCCESS! Indexed 1 text chunks to BigQuery
======================================================================
```

## Use Cases

1. **Documentation Indexing**: Store product documentation, user guides, FAQs
2. **Feature Descriptions**: Index UI feature explanations and walkthroughs
3. **Knowledge Base**: Build searchable knowledge bases
4. **Training Data**: Create training datasets for RAG systems
5. **Content Search**: Enable semantic search over text content

## Tips

- **Chunk Size**: Keep chunks between 100-1000 words for best results
- **Source Names**: Use descriptive source names for easy filtering
- **Metadata**: Add relevant metadata for better organization
- **Batch Processing**: Process multiple chunks at once for efficiency

## Successfully Indexed

✅ **XPress UPI Payment Guide** (1 chunk)
- Source: `xpress_upi_payment_guide`
- Category: `upi_payment`
- Dimensions: 768
- Status: Successfully stored in BigQuery

## Next Steps

1. Add more text chunks to `batch_index_text.py`
2. Run the script to index them
3. Use the embeddings for semantic search in your applications
4. Query the `vector_embeddings_new` table to retrieve relevant content
