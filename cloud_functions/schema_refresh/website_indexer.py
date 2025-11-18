"""
Website indexer module for Cloud Function
Scrapes websites and stores embeddings in BigQuery
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from bigquery_vector_store import BigQueryVectorStore
from typing import List


def scrape_website_content(url: str) -> Document:
    """
    Scrape content from a website including text and tables.

    Args:
        url: Website URL to scrape

    Returns:
        Document with page content and metadata
    """
    print(f"  Scraping: {url}")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header',
                           'iframe', 'noscript', 'aside', 'form', 'button']):
            element.decompose()

        # Try to find main content area
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content',
                        '#content', '.main-content', '#main-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body')

        content_parts = []

        # Extract title
        title = soup.title.string if soup.title else ""
        h1 = main_content.find('h1')
        if h1:
            title = h1.get_text(strip=True)
            content_parts.append(f"=== {title} ===\n")

        # Extract text with structure preservation
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                if element.name.startswith('h'):
                    content_parts.append(f"\n## {text} ##\n")
                else:
                    content_parts.append(text)

        # Extract tables
        tables = main_content.find_all('table')
        if tables:
            content_parts.append("\n\n=== TABLES ===")
            for idx, table in enumerate(tables, 1):
                table_text = f"\n--- Table {idx} ---\n"

                headers = []
                header_row = table.find('thead')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                    table_text += " | ".join(headers) + "\n"
                    table_text += "-" * (len(" | ".join(headers))) + "\n"

                rows = table.find_all('tr')
                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    if cells and cells != headers:
                        table_text += " | ".join(cells) + "\n"

                content_parts.append(table_text)

        # Combine all content
        full_content = "\n".join(content_parts)
        full_content = '\n'.join(line.strip() for line in full_content.split('\n') if line.strip())

        doc = Document(
            page_content=full_content,
            metadata={
                "source": url,
                "type": "website",
                "title": title,
                "url": url
            }
        )

        print(f"  ✓ Scraped {len(full_content)} characters from {url}")
        return doc

    except Exception as e:
        print(f"  ❌ Error scraping {url}: {e}")
        return None


def create_website_embeddings_internal(
    project_id: str,
    dataset: str,
    location: str = "us-central1",
    website_urls: List[str] = None
) -> dict:
    """
    Create and store website embeddings in BigQuery.

    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        location: GCP location
        website_urls: List of URLs to scrape and index

    Returns:
        Dict with success status and details
    """
    try:
        # Get URLs from environment or use default
        if not website_urls:
            urls_env = os.environ.get('WEBSITE_URLS', '')
            if urls_env:
                # Split on semicolons (used to avoid gcloud env var parsing issues)
                website_urls = [url.strip() for url in urls_env.split(';') if url.strip()]
            else:
                print("  ⚠️  No website URLs configured, skipping website indexing")
                return {
                    'success': True,
                    'chunks_indexed': 0,
                    'skipped': True,
                    'message': 'No website URLs configured'
                }

        print(f"  Processing {len(website_urls)} website(s)...")

        # Step 1: Scrape websites
        print("  Scraping websites...")
        documents = []
        for url in website_urls:
            doc = scrape_website_content(url)
            if doc:
                documents.append(doc)

        if not documents:
            return {
                'success': False,
                'error': 'No websites were successfully scraped'
            }

        print(f"  ✓ Scraped {len(documents)} website(s)")

        # Step 2: Split into chunks
        print("  Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=400,
            separators=[
                "\n=== ",
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
            length_function=len,
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

        print("  Creating embeddings for website chunks...")
        embeddings_data = []

        for idx, doc in enumerate(docs):
            embedding = embeddings_model.embed_query(doc.page_content)

            # Prepare metadata
            metadata = {
                "url": str(doc.metadata.get("url", "")),
                "title": str(doc.metadata.get("title", "")),
                "chunk_index": int(idx),
                "source": str(doc.metadata.get("source", ""))
            }

            embeddings_data.append({
                "id": f"website_chunk_{idx}",
                "source_name": f"{doc.metadata.get('title', 'website')}_chunk_{idx}",
                "content": doc.page_content,
                "embedding": embedding,
                "metadata": metadata
            })

            if (idx + 1) % 10 == 0:
                print(f"    ✓ Created {idx + 1}/{len(docs)} embeddings...")

        print(f"  ✓ Created {len(embeddings_data)} website chunk embeddings")

        # Step 4: Initialize BigQuery Vector Store and insert
        print("  Initializing BigQuery Vector Store...")
        vector_store = BigQueryVectorStore(dataset_name=dataset)
        vector_store.create_embeddings_table()

        print("  Inserting embeddings into BigQuery...")
        success = vector_store.insert_embeddings(
            embeddings_data=embeddings_data,
            source_type="website"
        )

        if success:
            return {
                'success': True,
                'chunks_indexed': len(embeddings_data),
                'websites_scraped': len(documents),
                'embedding_dimensions': len(embeddings_data[0]['embedding']) if embeddings_data else 0
            }
        else:
            return {
                'success': False,
                'error': 'Failed to insert embeddings'
            }

    except Exception as e:
        print(f"  ❌ Error in website indexing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }
