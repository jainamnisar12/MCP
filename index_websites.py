"""
Website Indexer - Scrapes and indexes website content into BigQuery
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from google.cloud import bigquery
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from bs4 import BeautifulSoup
import requests

# Import the vector store class
from index_new_pdf import NewBigQueryVectorStore

def index_websites():
    """Index websites from config.WEBSITE_URLS into BigQuery."""
    
    print("\n" + "="*60)
    print("🌐 INDEXING WEBSITES TO BIGQUERY")
    print("="*60)

    # Step 1: Check for URLs
    urls = getattr(config, 'WEBSITE_URLS', [])
    if not urls:
        print("❌ No WEBSITE_URLS found in config.py")
        return False
    
    print(f"Found {len(urls)} URLs to index:")
    for url in urls:
        print(f"  • {url}")

    # Step 2: Initialize Embeddings
    print("\n[1/4] Initializing Vertex AI embeddings...")
    try:
        embeddings_model = VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION if hasattr(config, 'GCP_LOCATION') else "us-central1"
        )
        print("✓ Vertex AI embeddings initialized")
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        return False

    # Step 3: Process each URL
    print("\n[2/4] Processing websites...")
    all_embeddings_data = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    for i, url in enumerate(urls):
        print(f"\nProcessing URL {i+1}/{len(urls)}: {url}")
        try:
            # Load content
            loader = WebBaseLoader(url)
            documents = loader.load()
            print(f"  ✓ Loaded content ({len(documents)} docs)")
            
            # Extract title from H1 tag using BeautifulSoup
            page_title = "Unknown Page"
            try:
                # Fetch the HTML content
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try to find H1 tag
                h1_tag = soup.find('h1')
                if h1_tag and h1_tag.get_text(strip=True):
                    page_title = h1_tag.get_text(strip=True)
                else:
                    # Fallback to HTML title
                    title_tag = soup.find('title')
                    if title_tag:
                        page_title = title_tag.get_text(strip=True)
            except Exception as e:
                print(f"  ⚠️ Could not extract title: {e}")
                # Use metadata title as last resort
                if documents and 'title' in documents[0].metadata:
                    page_title = documents[0].metadata['title'].strip()
            
            print(f"  ✓ Page title: {page_title}")
            
            # Split content
            chunks = text_splitter.split_documents(documents)
            print(f"  ✓ Split into {len(chunks)} chunks")
            
            # Create embeddings
            print(f"  Generating embeddings...")
            for idx, chunk in enumerate(chunks):
                try:
                    # Prepend title statement to chunk content
                    title_prefix = f"This is the {page_title} chunk.\n\n"
                    enhanced_content = title_prefix + chunk.page_content
                    
                    # Generate embedding for the enhanced content
                    embedding = embeddings_model.embed_query(enhanced_content)
                    
                    # Create unique ID
                    chunk_id = f"web_{i}_{idx}_{int(time.time())}"
                    
                    all_embeddings_data.append({
                        "id": chunk_id,
                        "source_name": url,
                        "content": enhanced_content,  # Store the enhanced content
                        "embedding": embedding,
                        "metadata": {
                            "url": url,
                            "title": page_title,
                            "chunk_index": idx,
                            "source": "website"
                        }
                    })
                except Exception as e:
                    print(f"  ⚠️ Error embedding chunk {idx}: {e}")
                    continue
            
            print(f"  ✓ Added {len(chunks)} embeddings for {url}")
            
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
            continue

    if not all_embeddings_data:
        print("❌ No embeddings generated")
        return False

    print(f"\n✓ Total embeddings generated: {len(all_embeddings_data)}")

    # Step 4: Log chunks instead of storing in BigQuery
    print("\n[3/4] Logging chunks (BigQuery insertion commented out)...")
    print("\n" + "="*80)
    print("📦 GENERATED CHUNKS:")
    print("="*80)
    
    for i, chunk_data in enumerate(all_embeddings_data):
        print(f"\n{'─'*80}")
        print(f"CHUNK {i+1}/{len(all_embeddings_data)}")
        print(f"{'─'*80}")
        print(f"ID: {chunk_data['id']}")
        print(f"Source: {chunk_data['source_name']}")
        print(f"Title: {chunk_data['metadata']['title']}")
        print(f"Chunk Index: {chunk_data['metadata']['chunk_index']}")
        print(f"Content Length: {len(chunk_data['content'])} chars")
        print(f"Embedding Dimensions: {len(chunk_data['embedding'])}")
        print(f"\nCONTENT:")
        print(f"{'-'*80}")
        print(chunk_data['content'])
        print(f"{'-'*80}")
    
    print(f"\n{'='*80}\n")

    # COMMENTED OUT: BigQuery insertion
    print("\n[3/4] Storing in BigQuery...")
    try:
        vector_store = NewBigQueryVectorStore(dataset_name=config.BIGQUERY_DATASET)
        
        # Ensure table exists (it should, but good to check)
        vector_store.create_embeddings_table()
        
        # Insert embeddings with source_type='website'
        success = vector_store.insert_embeddings(
            embeddings_data=all_embeddings_data,
            source_type="website"
        )
        
        if success:
            print("\n" + "="*60)
            print("✅ WEBSITES INDEXED SUCCESSFULLY!")
            print("="*60)
            return True
        else:
            print("❌ Failed to insert embeddings")
            return False
    
    except Exception as e:
        print(f"❌ Error storing in BigQuery: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✅ CHUNKS LOGGED SUCCESSFULLY (NOT PUSHED TO BIGQUERY)")
    print("="*60)
    return True

if __name__ == "__main__":
    success = index_websites()
    sys.exit(0 if success else 1)
