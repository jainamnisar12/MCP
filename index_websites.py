"""
Website Indexer v2 - Improved scraping with proper text extraction
Scrapes and indexes website content into BigQuery with clean text handling
"""

import os
import sys
import time
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from google.cloud import bigquery
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from bs4 import BeautifulSoup
import requests

# Import the vector store class
from index_new_pdf import NewBigQueryVectorStore


@dataclass
class Document:
    """Simple Document class for text chunks."""
    page_content: str
    metadata: Dict = field(default_factory=dict)


class WebContentExtractor:
    """Extracts clean, well-formatted content from web pages."""
    
    # Elements to remove (noise)
    NOISE_TAGS = [
        'script', 'style', 'noscript', 'iframe', 'svg', 
        'canvas', 'video', 'audio', 'map', 'object', 'embed'
    ]
    
    # Navigation/UI elements to remove
    NAV_TAGS = ['nav', 'header', 'footer', 'aside']
    
    # Classes/IDs that typically contain noise
    NOISE_PATTERNS = re.compile(
        r'nav|menu|sidebar|footer|header|cookie|banner|popup|modal|advertisement|social|share',
        re.IGNORECASE
    )
    
    def __init__(self, remove_navigation: bool = True):
        self.remove_navigation = remove_navigation
    
    def fetch_page(self, url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"  ⚠️ Error fetching {url}: {e}")
            return None
    
    def clean_soup(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove noise elements from the soup."""
        # Remove script, style, etc.
        for tag in soup(self.NOISE_TAGS):
            if tag:
                tag.decompose()
        
        # Remove navigation elements if specified
        if self.remove_navigation:
            for tag in soup(self.NAV_TAGS):
                if tag:
                    tag.decompose()
        
        # Remove elements with noise class/id patterns
        # Collect elements to remove first, then remove them
        elements_to_remove = []
        for element in soup.find_all(True):
            if element is None:
                continue
            try:
                classes = element.get('class', [])
                if classes is None:
                    classes = []
                classes_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
                element_id = element.get('id', '') or ''
                
                if self.NOISE_PATTERNS.search(classes_str) or self.NOISE_PATTERNS.search(element_id):
                    elements_to_remove.append(element)
            except (AttributeError, TypeError):
                continue
        
        # Now remove collected elements
        for element in elements_to_remove:
            try:
                element.decompose()
            except Exception:
                pass
        
        return soup
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title from H1 or title tag."""
        # Try H1 first
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(separator=' ', strip=True)
            if title:
                return title
        
        # Fallback to title tag
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean common suffixes like "| Company Name"
            title = re.split(r'\s*[|\-–—]\s*', title)[0].strip()
            return title
        
        return "Unknown Page"
    
    def fix_text_spacing(self, text: str) -> str:
        """Fix common text spacing issues from HTML extraction."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Add space between camelCase joins (e.g., "HomePayments" -> "Home Payments")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add space after punctuation if missing (e.g., "Hello.World" -> "Hello. World")
        text = re.sub(r'([.!?,:;])([A-Za-z])', r'\1 \2', text)
        
        # Fix common patterns like "Ltd.14th" -> "Ltd. 14th"
        text = re.sub(r'([a-zA-Z])\.(\d)', r'\1. \2', text)
        
        # Fix number-letter joins (e.g., "400078Maharashtra" -> "400078 Maharashtra")
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content area of the page."""
        # Try to find main content container
        main_selectors = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', {'role': 'main'}),
            soup.find('div', class_=re.compile(r'^(content|main|body)', re.I)),
            soup.find('div', id=re.compile(r'^(content|main|body)', re.I)),
        ]
        
        main_content = next((el for el in main_selectors if el), None)
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else ''
        
        return self.fix_text_spacing(text)
    
    def extract_structured_content(self, soup: BeautifulSoup) -> Dict:
        """Extract structured content (headings, paragraphs, lists, FAQs)."""
        content = {
            'sections': [],
            'faqs': [],
            'key_features': []
        }
        
        # Extract sections based on headings
        for heading in soup.find_all(['h2', 'h3']):
            heading_text = heading.get_text(separator=' ', strip=True)
            if not heading_text or len(heading_text) < 3:
                continue
            
            # Collect content until next heading
            section_content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3']:
                    break
                text = sibling.get_text(separator=' ', strip=True)
                if text and len(text) > 10:
                    section_content.append(text)
            
            if section_content:
                content['sections'].append({
                    'heading': self.fix_text_spacing(heading_text),
                    'content': self.fix_text_spacing(' '.join(section_content))
                })
        
        # Extract FAQs (common patterns)
        for faq_container in soup.find_all(True, class_=re.compile(r'faq', re.I)):
            questions = faq_container.find_all(['h3', 'h4', 'button', 'summary', 'strong'])
            for q in questions:
                q_text = q.get_text(separator=' ', strip=True)
                if '?' in q_text or q_text.lower().startswith(('what', 'how', 'why', 'when', 'who', 'is', 'can', 'does')):
                    # Find answer
                    answer_elem = q.find_next(['p', 'div', 'dd'])
                    if answer_elem:
                        a_text = answer_elem.get_text(separator=' ', strip=True)
                        if a_text and len(a_text) > 20:
                            content['faqs'].append({
                                'question': self.fix_text_spacing(q_text),
                                'answer': self.fix_text_spacing(a_text)
                            })
        
        return content
    
    def extract_all(self, url: str) -> Optional[Dict]:
        """Extract all content from a URL."""
        soup = self.fetch_page(url)
        if not soup:
            return None
        
        # Get title before cleaning
        title = self.extract_title(soup)
        
        # Clean the soup
        soup = self.clean_soup(soup)
        
        # Extract content
        main_content = self.extract_main_content(soup)
        structured = self.extract_structured_content(soup)
        
        return {
            'url': url,
            'title': title,
            'main_content': main_content,
            'sections': structured['sections'],
            'faqs': structured['faqs']
        }


def create_chunks_from_content(
    content: Dict,
    text_splitter: RecursiveCharacterTextSplitter
) -> List[Document]:
    """Create document chunks from extracted content."""
    chunks = []
    url = content['url']
    title = content['title']
    
    # Chunk the main content
    if content['main_content']:
        main_doc = Document(
            page_content=content['main_content'],
            metadata={
                'url': url,
                'title': title,
                'content_type': 'main',
                'source': 'website'
            }
        )
        chunks.extend(text_splitter.split_documents([main_doc]))
    
    # Create separate chunks for each section (if substantial)
    for section in content['sections']:
        if len(section['content']) > 100:
            section_text = f"{section['heading']}\n\n{section['content']}"
            section_doc = Document(
                page_content=section_text,
                metadata={
                    'url': url,
                    'title': title,
                    'section': section['heading'],
                    'content_type': 'section',
                    'source': 'website'
                }
            )
            # Don't split sections - keep them as coherent units if reasonable size
            if len(section_text) < 2000:
                chunks.append(section_doc)
            else:
                chunks.extend(text_splitter.split_documents([section_doc]))
    
    # Create separate chunks for FAQs (each Q&A as one chunk)
    for faq in content['faqs']:
        faq_text = f"Question: {faq['question']}\n\nAnswer: {faq['answer']}"
        faq_doc = Document(
            page_content=faq_text,
            metadata={
                'url': url,
                'title': title,
                'question': faq['question'],
                'content_type': 'faq',
                'source': 'website'
            }
        )
        chunks.append(faq_doc)
    
    return chunks


def index_websites():
    """Index websites from config.WEBSITE_URLS into BigQuery."""
    
    print("\n" + "="*60)
    print("🌐 INDEXING WEBSITES TO BIGQUERY (v2 - Improved)")
    print("="*60)

    # Step 1: Check for URLs
    urls = getattr(config, 'WEBSITE_URLS', [])
    if not urls:
        print("❌ No WEBSITE_URLS found in config.py")
        return False
    
    print(f"Found {len(urls)} URLs to index:")
    for url in urls:
        print(f"  • {url}")

    # Step 2: Initialize components
    print("\n[1/4] Initializing components...")
    
    try:
        embeddings_model = VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=config.GCP_PROJECT_ID,
            location=getattr(config, 'GCP_LOCATION', 'us-central1')
        )
        print("  ✓ Vertex AI embeddings initialized")
    except Exception as e:
        print(f"  ❌ Error initializing embeddings: {e}")
        return False
    
    extractor = WebContentExtractor(remove_navigation=True)
    print("  ✓ Web content extractor initialized")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunks for better retrieval
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    print("  ✓ Text splitter initialized")

    # Step 3: Process each URL
    print("\n[2/4] Processing websites...")
    all_embeddings_data = []

    for i, url in enumerate(urls):
        print(f"\n{'─'*50}")
        print(f"Processing URL {i+1}/{len(urls)}: {url}")
        print(f"{'─'*50}")
        
        try:
            # Extract content
            content = extractor.extract_all(url)
            if not content:
                print(f"  ❌ Failed to extract content")
                continue
            
            print(f"  ✓ Title: {content['title']}")
            print(f"  ✓ Main content: {len(content['main_content'])} chars")
            print(f"  ✓ Sections found: {len(content['sections'])}")
            print(f"  ✓ FAQs found: {len(content['faqs'])}")
            
            # Create chunks
            chunks = create_chunks_from_content(content, text_splitter)
            print(f"  ✓ Created {len(chunks)} chunks")
            
            # Generate embeddings
            print(f"  Generating embeddings...")
            for idx, chunk in enumerate(chunks):
                try:
                    # Prepend title context
                    title_prefix = f"Page: {content['title']}\n\n"
                    enhanced_content = title_prefix + chunk.page_content
                    
                    # Generate embedding
                    embedding = embeddings_model.embed_query(enhanced_content)
                    
                    # Create unique ID
                    content_type = chunk.metadata.get('content_type', 'main')
                    chunk_id = f"web_{i}_{content_type}_{idx}_{int(time.time())}"
                    
                    all_embeddings_data.append({
                        "id": chunk_id,
                        "source_name": url,
                        "content": enhanced_content,
                        "embedding": embedding,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_index": idx,
                        }
                    })
                    
                except Exception as e:
                    print(f"  ⚠️ Error embedding chunk {idx}: {e}")
                    continue
            
            print(f"  ✓ Generated {len(chunks)} embeddings")
            
        except Exception as e:
            print(f"  ❌ Error processing {url}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_embeddings_data:
        print("\n❌ No embeddings generated")
        return False

    print(f"\n✓ Total embeddings generated: {len(all_embeddings_data)}")

    # Step 4: Log chunks for verification
    print("\n[3/4] Chunk Preview...")
    print("\n" + "="*80)
    
    for i, chunk_data in enumerate(all_embeddings_data[:5]):  # Show first 5 only
        print(f"\n{'─'*80}")
        print(f"CHUNK {i+1} | Type: {chunk_data['metadata'].get('content_type', 'unknown')}")
        print(f"{'─'*80}")
        print(f"ID: {chunk_data['id']}")
        print(f"Source: {chunk_data['source_name']}")
        print(f"Title: {chunk_data['metadata'].get('title', 'N/A')}")
        print(f"Content Length: {len(chunk_data['content'])} chars")
        print(f"\nCONTENT PREVIEW (first 500 chars):")
        print(f"{'-'*40}")
        print(chunk_data['content'][:500] + "..." if len(chunk_data['content']) > 500 else chunk_data['content'])
        print(f"{'-'*40}")
    
    if len(all_embeddings_data) > 5:
        print(f"\n... and {len(all_embeddings_data) - 5} more chunks")
    
    print(f"\n{'='*80}\n")

    # Step 5: Store in BigQuery
    print("\n[4/4] Storing in BigQuery...")
    try:
        vector_store = NewBigQueryVectorStore(dataset_name=config.BIGQUERY_DATASET)
        
        # Ensure table exists
        vector_store.create_embeddings_table()
        
        # Insert embeddings
        success = vector_store.insert_embeddings(
            embeddings_data=all_embeddings_data,
            source_type="website"
        )
        
        if success:
            print("\n" + "="*60)
            print("✅ WEBSITES INDEXED SUCCESSFULLY!")
            print(f"   Total chunks: {len(all_embeddings_data)}")
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


def test_extraction(url: str):
    """Test content extraction for a single URL (no BigQuery)."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING EXTRACTION: {url}")
    print(f"{'='*60}\n")
    
    extractor = WebContentExtractor(remove_navigation=True)
    content = extractor.extract_all(url)
    
    if not content:
        print("❌ Failed to extract content")
        return
    
    print(f"📄 Title: {content['title']}")
    print(f"\n{'─'*60}")
    print("MAIN CONTENT (first 1000 chars):")
    print(f"{'─'*60}")
    print(content['main_content'][:1000])
    
    print(f"\n{'─'*60}")
    print(f"SECTIONS ({len(content['sections'])} found):")
    print(f"{'─'*60}")
    for i, section in enumerate(content['sections'][:3]):
        print(f"\n[{i+1}] {section['heading']}")
        print(f"    {section['content'][:200]}...")
    
    print(f"\n{'─'*60}")
    print(f"FAQs ({len(content['faqs'])} found):")
    print(f"{'─'*60}")
    for i, faq in enumerate(content['faqs'][:3]):
        print(f"\n[{i+1}] Q: {faq['question']}")
        print(f"    A: {faq['answer'][:150]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Index websites to BigQuery')
    parser.add_argument('--test', type=str, help='Test extraction for a single URL')
    args = parser.parse_args()
    
    if args.test:
        test_extraction(args.test)
    else:
        success = index_websites()
        sys.exit(0 if success else 1)