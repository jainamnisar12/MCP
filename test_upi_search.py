#!/usr/bin/env python3

"""
Test script for UPI document search with detailed logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from langchain_google_vertexai import VertexAIEmbeddings
from index_new_pdf import NewBigQueryVectorStore

def test_upi_search():
    print("🧪 Testing UPI Document Search")
    print("=" * 50)
    
    # Initialize embeddings
    print("1️⃣ Initializing embeddings...")
    embeddings = VertexAIEmbeddings(
        model_name="gemini-embedding-001",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION
    )
    print(f"✅ Embeddings model: {embeddings.model_name}")
    
    # Initialize vector store
    print("\n2️⃣ Initializing BigQuery vector store...")
    vector_store = NewBigQueryVectorStore(dataset_name=config.BIGQUERY_DATASET)
    print(f"✅ Vector store table: {vector_store.full_table_id}")
    
    # Test query
    question = "What is NPCI?"
    print(f"\n3️⃣ Testing search for: '{question}'")
    
    # Create embedding
    print("Creating query embedding...")
    query_embedding = embeddings.embed_query(question)
    print(f"Query embedding dimensions: {len(query_embedding)}")
    
    # Search
    print("Performing similarity search...")
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        k=3,
        similarity_threshold=0.3
    )
    
    print(f"\n4️⃣ Final Results Summary:")
    print(f"   • Total results: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"   • Result {i}: {result['source_name']} (score: {result['similarity_score']:.4f})")
        print(f"     Content: {result['content'][:100]}...")

if __name__ == "__main__":
    test_upi_search()
