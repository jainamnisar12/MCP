"""
Schema Cache Manager
Manages cached schema that refreshes every 7 days via cron job
"""

import config
import json
from pathlib import Path
from datetime import datetime
from google.cloud import bigquery
from typing import Dict, List, Optional
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate

# Cache configuration
CACHE_DIR = Path("./schema_cache")
CACHE_FILE = CACHE_DIR / "schema_cache.json"
CACHE_VALIDITY_DAYS = 7


class SchemaCache:
    """Manages schema caching with automatic refresh"""
    
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache_dir = cache_file.parent
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_cache(self, schema_info: Dict, table_contexts: Dict):
        """
        Save schema and contexts to cache file with timestamp
        
        Args:
            schema_info: Schema information from BigQuery
            table_contexts: Generated table contexts
        """
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "schema_info": schema_info,
            "table_contexts": table_contexts,
            "project_id": config.GCP_PROJECT_ID,
            "dataset": config.BIGQUERY_DATASET
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        print(f"✓ Schema cache saved to: {self.cache_file}")
        print(f"  Timestamp: {cache_data['timestamp']}")
        print(f"  Tables: {len(schema_info)}")
    
    def load_cache(self) -> Optional[Dict]:
        """
        Load cached schema if it exists and is valid
        
        Returns:
            Cached data dict or None if cache is invalid/missing
        """
        if not self.cache_file.exists():
            print(f"⚠️  No cache file found at: {self.cache_file}")
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check cache age
            cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
            age_days = (datetime.now() - cache_timestamp).days
            
            print(f"\n{'='*60}")
            print(f"📦 LOADING CACHED SCHEMA")
            print(f"{'='*60}")
            print(f"Cache file: {self.cache_file}")
            print(f"Cache age: {age_days} days")
            print(f"Created: {cache_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Tables: {len(cache_data['schema_info'])}")
            
            if age_days > CACHE_VALIDITY_DAYS:
                print(f"⚠️  Cache is stale (>{CACHE_VALIDITY_DAYS} days old)")
                print(f"{'='*60}\n")
                return None
            
            print(f"✓ Cache is valid")
            print(f"{'='*60}\n")
            
            return cache_data
            
        except Exception as e:
            print(f"❌ Error loading cache: {str(e)}")
            return None
    
    def is_cache_valid(self) -> bool:
        """Check if cache exists and is still valid"""
        if not self.cache_file.exists():
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
            age_days = (datetime.now() - cache_timestamp).days
            
            return age_days <= CACHE_VALIDITY_DAYS
        except:
            return False
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache"""
        if not self.cache_file.exists():
            return {
                "exists": False,
                "valid": False
            }
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
            age_days = (datetime.now() - cache_timestamp).days
            valid = age_days <= CACHE_VALIDITY_DAYS
            
            return {
                "exists": True,
                "valid": valid,
                "timestamp": cache_data['timestamp'],
                "age_days": age_days,
                "tables_count": len(cache_data['schema_info']),
                "project_id": cache_data.get('project_id'),
                "dataset": cache_data.get('dataset')
            }
        except Exception as e:
            return {
                "exists": True,
                "valid": False,
                "error": str(e)
            }


def fetch_bigquery_schema(client: bigquery.Client, dataset_name: str) -> Dict[str, List[Dict]]:
    """
    Fetch the current schema from BigQuery for all tables in a dataset.
    
    Returns:
        Dict mapping table names to their schema information
    """
    print(f"\n{'='*60}")
    print(f"📊 FETCHING SCHEMA FROM BIGQUERY")
    print(f"{'='*60}")
    print(f"Project: {config.GCP_PROJECT_ID}")
    print(f"Dataset: {dataset_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    schema_info = {}
    
    try:
        dataset_ref = f"{config.GCP_PROJECT_ID}.{dataset_name}"
        
        # List all tables in the dataset
        tables = list(client.list_tables(dataset_ref))
        print(f"Found {len(tables)} table(s) in dataset:\n")
        
        for table_item in tables:
            table_name = table_item.table_id
            table_ref = f"{dataset_ref}.{table_name}"
            
            print(f"  📋 Fetching schema for: {table_name}")
            
            # Get table schema
            table = client.get_table(table_ref)
            
            schema_fields = []
            for field in table.schema:
                schema_fields.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or ""
                })
            
            schema_info[table_name] = {
                "fields": schema_fields,
                "num_rows": table.num_rows,
                "created": str(table.created),
                "modified": str(table.modified)
            }
            
            print(f"     ✓ {len(schema_fields)} columns, {table.num_rows} rows")
        
        print(f"\n{'='*60}")
        print(f"✓ Schema fetch completed successfully!")
        print(f"{'='*60}\n")
        
        return schema_info
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR FETCHING SCHEMA")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(f"{'='*60}\n")
        raise


def generate_intelligent_fallback(
    table_name: str,
    schema_fields: List[Dict],
    num_rows: int
) -> Dict[str, str]:
    """
    Generate intelligent fallback context based on table structure analysis.
    
    Args:
        table_name: Name of the table
        schema_fields: List of field dictionaries
        num_rows: Number of rows in the table
    
    Returns:
        Dict with generated table context
    """
    # Analyze field names to understand table purpose
    field_names = [f['name'].lower() for f in schema_fields]
    
    # Detect table type patterns
    has_id = any('id' in name for name in field_names)
    has_name = any('name' in name for name in field_names)
    has_email = any('email' in name for name in field_names)
    has_phone = any('phone' in name for name in field_names)
    has_address = any('address' in name for name in field_names)
    has_timestamp = any('timestamp' in name or 'date' in name or 'time' in name for name in field_names)
    has_amount = any('amount' in name or 'price' in name or 'cost' in name for name in field_names)
    has_status = any('status' in name for name in field_names)
    
    # Build field list for descriptions
    key_fields = [f['name'] for f in schema_fields[:5]]  # First 5 fields
    field_list = ", ".join(key_fields)
    if len(schema_fields) > 5:
        field_list += f", and {len(schema_fields) - 5} more"
    
    # Determine if likely contains sensitive data
    is_sensitive = has_email or has_phone or has_address or 'ssn' in ' '.join(field_names)
    
    # Generate context based on patterns
    table_type = "general data"
    if has_email and has_phone and has_name:
        table_type = "contact/customer information"
    elif has_amount and has_timestamp:
        table_type = "transactional records"
    elif has_status and has_timestamp:
        table_type = "operational tracking"
    elif has_name and has_id:
        table_type = "master data/reference"
    
    # Get primary key field (usually first field with 'id' in name)
    primary_key = next((f['name'] for f in schema_fields if 'id' in f['name'].lower()), schema_fields[0]['name'])
    
    # Build description
    description = (
        f"The {table_name} table stores {table_type} with {num_rows:,} records. "
        f"Key fields include {field_list}. "
        f"This table contains {len(schema_fields)} columns providing comprehensive data for {table_name.replace('_', ' ')} management."
    )
    
    # Build usage
    usage = (
        f"Query this table to retrieve and analyze {table_name.replace('_', ' ')} data. "
        f"Use for lookups by {primary_key}, reporting, and joining with related tables for comprehensive analysis."
    )
    
    # Build business context
    business_function = "data management and reporting"
    if has_amount:
        business_function = "financial tracking and analysis"
    elif has_status:
        business_function = "operational monitoring and workflow management"
    elif has_email or has_phone:
        business_function = "customer relationship management and communications"
    
    business_context = f"Supports {business_function} for {table_name.replace('_', ' ')}-related business processes."
    
    # Generate sample queries
    query1 = f"SELECT * FROM {table_name} WHERE {primary_key} = 'X'"
    query2 = f"SELECT COUNT(*) FROM {table_name}"
    if has_timestamp:
        timestamp_field = next(f['name'] for f in schema_fields if 'timestamp' in f['name'].lower() or 'date' in f['name'].lower())
        query2 = f"SELECT * FROM {table_name} WHERE {timestamp_field} > '2024-01-01' ORDER BY {timestamp_field} DESC"
    
    common_queries = f"{query1}; {query2}"
    
    return {
        "description": description,
        "usage": usage,
        "business_context": business_context,
        "common_queries": common_queries,
        "sensitive": is_sensitive,
        "row_level_security": is_sensitive
    }


def generate_table_context_with_gemini(
    table_name: str, 
    schema_fields: List[Dict], 
    num_rows: int,
    llm: ChatVertexAI
) -> Dict[str, str]:
    """
    Use Gemini via Vertex AI to generate intelligent context for a table.
    
    Args:
        table_name: Name of the table
        schema_fields: List of field dictionaries
        num_rows: Number of rows in the table
        llm: Initialized Vertex AI Gemini LLM instance
    
    Returns:
        Dict with table context information
    """
    import json
    
    fields_str = "\n".join([
        f"  - {field['name']} ({field['type']}) {'[REQUIRED]' if field['mode'] == 'REQUIRED' else ''}"
        for field in schema_fields
    ])
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database documentation expert. Analyze the table schema and generate helpful context.

Return ONLY a valid JSON object with these fields (no markdown, no backticks, no extra text):
{{
    "description": "Write 2-3 clear sentences explaining what data this table stores, what it's used for, and its key characteristics. Be specific based on the column names.",
    "usage": "Write 2 sentences about when developers should query this table and what business questions it helps answer.",
    "business_context": "Write 1 sentence about what business function or process this table supports.",
    "common_queries": "Provide 1-2 realistic SQL query examples based on the actual columns in this table.",
    "sensitive": "true or false"
}}

Important: Base your analysis on the actual table name and column names provided. Be specific, not generic."""),
        ("human", """Analyze this table:

Table: {table_name}
Rows: {num_rows}

Columns:
{fields}

Generate detailed context in pure JSON format (no markdown).""")
    ])
    
    try:
        response = (context_prompt | llm).invoke({
            "table_name": table_name,
            "num_rows": num_rows,
            "fields": fields_str
        })
        
        content = response.content.strip()
        
        print(f"     📝 Gemini response length: {len(content)} chars")
        
        # Remove markdown code blocks if present
        content = content.replace('```json', '').replace('```', '').strip()
        
        # Extract JSON from response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            context_json = json.loads(json_str)
            
            # Validate we got real content, not generic
            desc = context_json.get("description", "")
            if "core data records" in desc.lower() or len(desc) < 50:
                print(f"     ⚠️  Generic response detected (length: {len(desc)})")
                raise ValueError("Generated generic context, using fallback...")
            
            print(f"     ✅ Gemini context validated and accepted")
            
            # Handle sensitive field as either boolean or string
            sensitive_value = context_json.get("sensitive", "true")
            if isinstance(sensitive_value, bool):
                is_sensitive = sensitive_value
            else:
                is_sensitive = str(sensitive_value).lower() == "true"

            return {
                "description": context_json.get("description", ""),
                "usage": context_json.get("usage", ""),
                "business_context": context_json.get("business_context", ""),
                "common_queries": context_json.get("common_queries", ""),
                "sensitive": is_sensitive,
                "row_level_security": is_sensitive
            }
        else:
            raise ValueError("Could not extract JSON from response")
        
    except json.JSONDecodeError as je:
        print(f"     ⚠️  JSON parsing error: {str(je)}")
        print(f"     Raw response preview: {content[:200] if 'content' in locals() else 'N/A'}...")
        print(f"     Using intelligent fallback context...")
    except Exception as e:
        print(f"     ⚠️  Gemini generation failed: {type(e).__name__}: {str(e)}")
        print(f"     Using intelligent fallback context...")
    
    # Generate intelligent fallback based on table analysis
    return generate_intelligent_fallback(table_name, schema_fields, num_rows)


def generate_all_table_contexts(
    schema_info: Dict, 
    llm: ChatVertexAI
) -> Dict[str, Dict]:
    """
    Generate context for all tables using Gemini.
    
    Args:
        schema_info: Schema information from fetch_bigquery_schema
        llm: Initialized Vertex AI Gemini LLM instance
    
    Returns:
        Dict mapping table names to their generated contexts
    """
    print(f"\n{'='*60}")
    print(f"🤖 GENERATING TABLE CONTEXTS WITH GEMINI (Vertex AI)")
    print(f"{'='*60}\n")
    
    table_contexts = {}
    
    for table_name, info in schema_info.items():
        print(f"  🧠 Generating context for: {table_name}")
        
        context = generate_table_context_with_gemini(
            table_name=table_name,
            schema_fields=info["fields"],
            num_rows=info["num_rows"],
            llm=llm
        )
        
        table_contexts[table_name] = context
        
        print(f"     ✓ {context['description'][:80]}...")
    
    print(f"\n{'='*60}")
    print(f"✓ All table contexts generated!")
    print(f"{'='*60}\n")
    
    return table_contexts


def format_schema_for_llm(schema_info: Dict, table_contexts: Dict) -> str:
    """
    Format the schema information into a readable string for the LLM.
    
    Args:
        schema_info: Schema information from fetch_bigquery_schema
        table_contexts: Table context information
    
    Returns:
        Formatted schema string
    """
    schema_parts = []
    
    schema_parts.append("=" * 60)
    schema_parts.append("DATABASE SCHEMA (Cached)")
    schema_parts.append("=" * 60)
    
    for table_name, info in schema_info.items():
        context = table_contexts.get(table_name, {})
        
        schema_parts.append(f"\n📊 TABLE: {table_name.upper()}")
        schema_parts.append("-" * 60)
        
        if context.get("description"):
            schema_parts.append(f"📝 Description: {context['description']}")
        
        if context.get("business_context"):
            schema_parts.append(f"💼 Business Context: {context['business_context']}")
        
        if context.get("usage"):
            schema_parts.append(f"🎯 Usage: {context['usage']}")
        
        if context.get("common_queries"):
            schema_parts.append(f"📋 Common Queries: {context['common_queries']}")
        
        schema_parts.append(f"📊 Rows: {info['num_rows']:,}")
        
        if context.get("sensitive"):
            schema_parts.append(f"🔒 Sensitive Data: Yes - Row-level security enforced")
        
        schema_parts.append("\n📋 Columns:")
        
        for field in info["fields"]:
            field_desc = f"  • {field['name']} ({field['type']})"
            if field['mode'] == 'REQUIRED':
                field_desc += " [REQUIRED]"
            if field['description']:
                field_desc += f" - {field['description']}"
            schema_parts.append(field_desc)
        
        schema_parts.append("")
    
    schema_parts.append("=" * 60)
    
    return "\n".join(schema_parts)


def load_or_refresh_schema(
    bq_client: bigquery.Client,
    llm: ChatVertexAI,
    force_refresh: bool = False
) -> tuple[Dict, Dict, str]:
    """
    Load schema from cache or refresh if needed.
    
    Args:
        bq_client: BigQuery client
        llm: Vertex AI LLM instance
        force_refresh: Force refresh even if cache is valid
    
    Returns:
        Tuple of (schema_info, table_contexts, formatted_schema)
    """
    cache_manager = SchemaCache()
    
    # Check if we should use cache
    if not force_refresh and cache_manager.is_cache_valid():
        cache_data = cache_manager.load_cache()
        if cache_data:
            return (
                cache_data['schema_info'],
                cache_data['table_contexts'],
                format_schema_for_llm(
                    cache_data['schema_info'],
                    cache_data['table_contexts']
                )
            )
    
    # Cache invalid or force refresh - fetch fresh data
    print("🔄 Refreshing schema from BigQuery...")
    schema_info = fetch_bigquery_schema(bq_client, config.BIGQUERY_DATASET)
    table_contexts = generate_all_table_contexts(schema_info, llm)
    
    # Save to cache
    cache_manager.save_cache(schema_info, table_contexts)
    
    formatted_schema = format_schema_for_llm(schema_info, table_contexts)
    
    return schema_info, table_contexts, formatted_schema