import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

# --- Performance Metrics Logger ---
perf_logger = logging.getLogger('performance_metrics')
perf_logger.setLevel(logging.INFO)

perf_handler = logging.FileHandler('performance_metrics.log')
perf_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
perf_logger.addHandler(perf_handler)

@dataclass
class PerformanceMetrics:
    """Store performance metrics for a single request"""
    timestamp: str
    user: str
    user_type: str
    query: str

    # Timing metrics (in seconds)
    total_time: float
    sql_generation_time: Optional[float] = None
    sql_execution_time: Optional[float] = None
    output_formatting_time: Optional[float] = None
    pdf_search_time: Optional[float] = None
    agent_response_time: Optional[float] = None

    # Query metrics
    rows_returned: Optional[int] = None
    bytes_processed: Optional[int] = None
    cache_hit: bool = False

    # Tool usage
    tools_used: list = None
    tool_count: int = 0

    # Status
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class PerformanceTracker:
    """Track and aggregate performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.session_stats = defaultdict(lambda: {
            'total_queries': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'sql_queries': 0,
            'pdf_queries': 0,
            'errors': 0
        })
    
    def log_metric(self, metric: PerformanceMetrics):
        """Log a performance metric"""
        self.metrics_history.append(metric)
        
        # Update session stats
        user_stats = self.session_stats[metric.user]
        user_stats['total_queries'] += 1
        user_stats['total_time'] += metric.total_time
        user_stats['avg_time'] = user_stats['total_time'] / user_stats['total_queries']
        
        if metric.sql_execution_time:
            user_stats['sql_queries'] += 1
        if metric.pdf_search_time:
            user_stats['pdf_queries'] += 1
        if metric.status == "ERROR":
            user_stats['errors'] += 1
        
        # Log to file
        perf_logger.info(metric.to_json())
    
    def get_session_summary(self, user: str) -> Dict[str, Any]:
        """Get summary statistics for a user session"""
        stats = self.session_stats[user]
        
        # Calculate additional metrics from history
        user_metrics = [m for m in self.metrics_history if m.user == user]
        
        if user_metrics:
            sql_times = [m.sql_execution_time for m in user_metrics if m.sql_execution_time]
            pdf_times = [m.pdf_search_time for m in user_metrics if m.pdf_search_time]
            
            stats['sql_avg_time'] = sum(sql_times) / len(sql_times) if sql_times else 0
            stats['pdf_avg_time'] = sum(pdf_times) / len(pdf_times) if pdf_times else 0
            stats['min_response_time'] = min(m.total_time for m in user_metrics)
            stats['max_response_time'] = max(m.total_time for m in user_metrics)
        
        return stats
    
    def print_summary(self, user: str):
        """Print formatted session summary"""
        stats = self.get_session_summary(user)
        
        print("\n" + "="*60)
        print("📊 SESSION PERFORMANCE SUMMARY")
        print("="*60)
        print(f"User: {user}")
        print(f"\n📈 Query Statistics:")
        print(f"   • Total Queries: {stats['total_queries']}")
        print(f"   • SQL Queries: {stats['sql_queries']}")
        print(f"   • PDF Queries: {stats['pdf_queries']}")
        print(f"   • Errors: {stats['errors']}")
        
        print(f"\n⏱️  Response Times:")
        print(f"   • Average: {stats['avg_time']:.3f}s")
        if 'min_response_time' in stats:
            print(f"   • Fastest: {stats['min_response_time']:.3f}s")
            print(f"   • Slowest: {stats['max_response_time']:.3f}s")
        
        if stats['sql_queries'] > 0 and 'sql_avg_time' in stats:
            print(f"   • Avg SQL Execution: {stats['sql_avg_time']:.3f}s")
        if stats['pdf_queries'] > 0 and 'pdf_avg_time' in stats:
            print(f"   • Avg PDF Search: {stats['pdf_avg_time']:.3f}s")
        
        print(f"\n⏰ Total Session Time: {stats['total_time']:.3f}s")
        print("="*60)

# Global performance tracker instance
performance_tracker = PerformanceTracker()


# --- Performance Monitoring Decorator ---
def monitor_performance(func_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                print(f"⏱️  {func_name}: {execution_time:.3f}s")
                return result, execution_time
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"⏱️  {func_name}: {execution_time:.3f}s (ERROR)")
                raise e
        return wrapper
    return decorator


# --- Usage Example for agent.py ---
"""
Add this to your agent.py main loop:

while True:
    try:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            # Print performance summary before exit
            performance_tracker.print_summary(CURRENT_USER)
            break
        
        if not user_input:
            continue
        
        conversation_count += 1
        
        print("\nAssistant: ", end="", flush=True)
        
        # Start performance tracking
        start_time = time.time()
        metric = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            user=CURRENT_USER,
            user_type=USER_TYPE,
            query=user_input[:200]  # Truncate long queries
        )
        
        try:
            # Run the agent
            agent_start = time.time()
            response = root_agent.run(user_input)
            metric.agent_response_time = time.time() - agent_start
            
            print(response)
            print()
            
            # Calculate total time
            metric.total_time = time.time() - start_time
            metric.status = "SUCCESS"
            
            # Log the metric
            performance_tracker.log_metric(metric)
            
            # Print performance summary for this query
            print(f"\n⏱️  Query completed in {metric.total_time:.3f}s")
            
        except Exception as agent_error:
            metric.total_time = time.time() - start_time
            metric.status = "ERROR"
            metric.error_message = str(agent_error)
            
            performance_tracker.log_metric(metric)
            
            print(f"\n⚠️ I encountered an issue processing your request.")
            print(f"Error details: {str(agent_error)}")
            print(f"Time taken: {metric.total_time:.3f}s")
            
    except KeyboardInterrupt:
        performance_tracker.print_summary(CURRENT_USER)
        break
"""


# --- Usage Example for mcp_toolbox_server.py ---
"""
Add this to your tool functions:

@mcp.tool
def query_customer_database(natural_language_query: str, current_user: str = None, user_type: str = 'customer') -> str:
    # Start tracking
    start_time = time.time()
    sql_gen_time = None
    sql_exec_time = None
    
    print(f"\n{'='*60}")
    print(f"[BQ Tool] Query: {natural_language_query}")
    print(f"[BQ Tool] User: {current_user or 'NONE'} ({user_type.upper()})")
    print(f"{'='*60}")
    
    # ... authentication and rate limiting checks ...
    
    # Track SQL generation time
    sql_start = time.time()
    sql_response = sql_generation_chain.invoke({
        "question": natural_language_query,
        "current_user": f"'{current_user}'",
        "user_type": user_type
    })
    sql_gen_time = time.time() - sql_start
    print(f"⏱️  SQL Generation: {sql_gen_time:.3f}s")

    sql_query = sql_response.content.strip()

    # ... validation checks ...

    # Track SQL execution time
    sql_exec_start = time.time()
    text_result, df_result = _execute_query(sql_query, current_user, user_type)
    sql_exec_time = time.time() - sql_exec_start
    print(f"⏱️  SQL Execution: {sql_exec_time:.3f}s")

    # Track output formatting time
    format_start = time.time()
    # ... build response string ...
    formatted_response = build_response(df_result, sql_query)
    output_format_time = time.time() - format_start
    print(f"⏱️  Output Formatting: {output_format_time:.3f}s")

    # Calculate total time
    total_time = time.time() - start_time

    # Log performance metric
    metric = PerformanceMetrics(
        timestamp=datetime.now().isoformat(),
        user=current_user or 'UNKNOWN',
        user_type=user_type,
        query=natural_language_query[:200],
        total_time=total_time,
        sql_generation_time=sql_gen_time,
        sql_execution_time=sql_exec_time,
        output_formatting_time=output_format_time,
        rows_returned=len(df_result) if df_result is not None else 0,
        tools_used=['query_customer_database'],
        tool_count=1,
        status='SUCCESS' if df_result is not None else 'ERROR'
    )

    perf_logger.info(metric.to_json())

    print(f"⏱️  Total Tool Time: {total_time:.3f}s")
    print(f"{'='*60}\n")

    # ... return response ...
"""