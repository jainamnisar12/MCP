import os
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from urllib.parse import urljoin, urlparse
import config
from typing import List
from google.cloud import vision

def extract_image_text(img_url: str, page_url: str) -> str:
    """
    Extract text description from images using Google Cloud Vision API.

    Args:
        img_url: URL of the image
        page_url: Base URL of the page (for resolving relative URLs)

    Returns:
        Text description of the image
    """
    try:
        # Resolve relative URLs
        full_img_url = urljoin(page_url, img_url)

        # Download image
        response = requests.get(full_img_url, timeout=10)
        if response.status_code != 200:
            return ""

        # Use Google Cloud Vision API
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=response.content)

        # Get text detection (OCR)
        text_response = client.text_detection(image=image)
        texts = text_response.text_annotations
        ocr_text = texts[0].description if texts else ""

        # Get label detection (image understanding)
        label_response = client.label_detection(image=image)
        labels = [label.description for label in label_response.label_annotations[:5]]

        # Combine OCR text and labels
        description = ""
        if ocr_text:
            description += f"Text in image: {ocr_text.strip()}\n"
        if labels:
            description += f"Image contains: {', '.join(labels)}"

        return description.strip()

    except Exception as e:
        print(f"  ⚠ Warning: Could not process image {img_url}: {e}")
        return ""

def scrape_website_content(url: str) -> Document:
    """
    Scrape content from a website including text, tables, and images.
    """
    print(f"\n🌐 Scraping: {url}")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements MORE AGGRESSIVELY
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'iframe', 'noscript', 'aside', 'form', 'button']):
            element.decompose()

        # Try to find main content area first
        main_content = None
        
        # Common main content selectors
        for selector in ['main', 'article', '[role="main"]', '.content', 
                        '#content', '.main-content', '#main-content']:
            main_content = soup.select_one(selector)
            if main_content:
                print(f"  ✓ Found main content using selector: {selector}")
                break
        
        # If no main content found, use body but be more selective
        if not main_content:
            main_content = soup.find('body')
            print("  ⚠ Using body content (no main container found)")

        content_parts = []

        # Extract title/heading
        title = soup.title.string if soup.title else ""
        h1 = main_content.find('h1')
        if h1:
            title = h1.get_text(strip=True)
            content_parts.append(f"=== {title} ===\n")

        # 1. Extract text with better structure preservation
        print("  📝 Extracting text...")
        
        # Get all paragraphs and headings
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
            text = element.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short text
                if element.name.startswith('h'):
                    content_parts.append(f"\n## {text} ##\n")
                else:
                    content_parts.append(text)

        # 2. Extract tables (keep your existing table extraction)
        print("  📊 Extracting tables...")
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
            print(f"  ✓ Extracted {len(tables)} table(s)")

        # Combine all content
        full_content = "\n".join(content_parts)
        
        # Clean up extra whitespace
        full_content = '\n'.join(line.strip() for line in full_content.split('\n') if line.strip())

        doc = Document(
            page_content=full_content,
            metadata={
                "source": url,
                "type": "website",
                "title": title,
                "url": url  # Add URL to metadata for better tracking
            }
        )

        print(f"  ✓ Successfully scraped {len(full_content)} characters")
        return doc

    except Exception as e:
        print(f"  ❌ Error scraping {url}: {e}")
        return None
    
def create_website_vector_store(urls: List[str], output_path: str = None):
    """
    Create FAISS vector store from website URLs using Vertex AI embeddings.
    """
    if output_path is None:
        output_path = config.WEBSITE_VECTOR_STORE_PATH

    print("\n" + "="*60)
    print("🌐 WEBSITE INDEXING WITH VERTEX AI")
    print("="*60)
    print(f"📊 URLs to process: {len(urls)}")

    # Step 1: Scrape all websites
    print("\n[1/4] Scraping websites...")
    documents = []
    for url in urls:
        doc = scrape_website_content(url)
        if doc:
            documents.append(doc)

    if not documents:
        print("❌ No documents were successfully scraped")
        return

    print(f"✓ Scraped {len(documents)} website(s)")

    # Step 2: Split into semantic chunks with good overlap
    print("\n[2/4] Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,  # INCREASED for more complete context
        chunk_overlap=400,  # INCREASED for better continuity
        separators=[
            "\n=== ",     # Main section markers
            "\n## ",      # Headers
            "\n### ",     # Subheaders
            "\n\n",       # Paragraphs
            "\n",         # Lines
            ". ",         # Sentences
            " ",          # Words
            ""            # Characters
        ],
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)
    print(f"✓ Created {len(docs)} text chunks (avg {sum(len(d.page_content) for d in docs)//len(docs)} chars/chunk)")
    
    # Print sample chunks for debugging
    print("\n📝 Sample chunks:")
    for i, doc in enumerate(docs[:3]):
        print(f"\nChunk {i+1} (from {doc.metadata['source']}):")
        print(f"{doc.page_content[:200]}...")

    # Step 3: Create embeddings
    print("\n[3/4] Creating embeddings with Vertex AI...")
    try:
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION if hasattr(config, 'GCP_LOCATION') else "us-central1"
        )
        print("✓ Vertex AI embeddings initialized")
    except Exception as e:
        print(f"❌ Error initializing Vertex AI embeddings: {e}")
        return

    # Step 4: Create or update FAISS vector store
    print(f"\n[4/4] Creating FAISS vector store at {output_path}...")
    try:
        os.makedirs(output_path, exist_ok=True)

        db = FAISS.from_documents(docs, embeddings)
        db.save_local(output_path)

        print("\n" + "="*60)
        print("✅ WEBSITE VECTOR STORE CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Location: {output_path}")
        print(f"📊 Total chunks: {len(docs)}")
        print(f"🌐 Websites indexed: {len(documents)}")
        print(f"🔧 Embedding model: text-embedding-004 (Vertex AI)")
        print("="*60 + "\n")

        return db

    except Exception as e:
        print(f"\n❌ Error creating vector store: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys

    # Check if URLs are provided in config
    if not config.WEBSITE_URLS:
        print("\n" + "="*60)
        print("🚀 WEBSITE INDEXER")
        print("="*60)
        print("\n⚠️  No URLs configured!")
        print("\nTo index websites, add URLs to config.py:")
        print("\nWEBSITE_URLS = [")
        print('    "https://example.com/page1",')
        print('    "https://example.com/page2",')
        print("]")
        print("\nThen run: python -m agent.website_indexer")
        print("\nOr provide URLs as command-line arguments:")
        print("python -m agent.website_indexer https://example.com/page1 https://example.com/page2")
        print("="*60 + "\n")
        sys.exit(1)

    # Use URLs from config or command line
    urls_to_index = sys.argv[1:] if len(sys.argv) > 1 else config.WEBSITE_URLS

    print("\n📋 URLs to index:")
    for url in urls_to_index:
        print(f"  • {url}")
    print()

    # Create the vector store
    create_website_vector_store(urls_to_index)

    print("\n✅ Done! The MCP server will automatically load this vector store.")
    print("   You can now query websites using: ask_document(question, source='website')")
