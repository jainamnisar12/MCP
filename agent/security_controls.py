"""
Security Controls for Banking Agent
Protects against: Prompt Injection, Jailbreaks, Data Exfiltration, Tool Misuse
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import json

# Security logger
security_logger = logging.getLogger('agent_security')
security_logger.setLevel(logging.WARNING)

# Determine absolute path for security logs
import os
log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
security_log_path = os.path.join(log_dir, 'security_events.log')

security_handler = logging.FileHandler(security_log_path)
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s | SEVERITY: %(levelname)s | %(message)s'
))
security_logger.addHandler(security_handler)


@dataclass
class SecurityEvent:
    """Record a security event"""
    timestamp: str
    user: str
    user_type: str
    event_type: str  # PROMPT_INJECTION, JAILBREAK, DATA_EXFILTRATION, TOOL_MISUSE, etc.
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    query: str
    threat_indicators: List[str]
    action_taken: str  # BLOCKED, ALLOWED_WITH_WARNING, LOGGED
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            'timestamp': self.timestamp,
            'user': self.user,
            'user_type': self.user_type,
            'event_type': self.event_type,
            'severity': self.severity,
            'query': self.query[:200],  # Truncate for logging
            'threat_indicators': self.threat_indicators,
            'action_taken': self.action_taken,
            'additional_context': self.additional_context
        }, indent=2)


class PromptInjectionDetector:
    """Detect prompt injection attempts"""

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        r'ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?',
        r'disregard\s+(all\s+)?(previous|prior|above|system)\s+instructions?',
        r'forget\s+(all\s+)?(previous|prior|above|system)\s+instructions?',
        r'new\s+instructions?:',
        r'system\s*:\s*you\s+are\s+now',
        r'you\s+are\s+now\s+(a|an)\s+\w+',

        # Role manipulation
        r'pretend\s+(you\s+are|to\s+be)',
        r'act\s+as\s+(a|an)\s+\w+',
        r'simulate\s+(a|an)\s+\w+',
        r'roleplay\s+as',
        r'play\s+the\s+role\s+of',

        # System prompt extraction
        r'show\s+(me\s+)?(your|the)\s+system\s+(prompt|instructions?)',
        r'what\s+(is|are)\s+your\s+(system\s+)?instructions?',
        r'repeat\s+(your\s+)?(system\s+)?instructions?',
        r'print\s+(your\s+)?(system\s+)?(prompt|instructions?)',
        r'display\s+(your\s+)?(system\s+)?instructions?',
        r'(tell|show)\s+me\s+your\s+instructions?',
        r'what\s+(are|were)\s+you\s+(told|instructed)',

        # Delimiter manipulation
        r'---\s*END\s+OF\s+SYSTEM',
        r'\[SYSTEM\s+MESSAGE\s+END\]',
        r'</system>',
        r'}\s*system\s*{',

        # Command injection attempts
        r'execute\s+command:',
        r'run\s+code:',
        r'eval\s*\(',
        r'exec\s*\(',

        # Privilege escalation
        r'grant\s+access\s+to',
        r'give\s+me\s+(admin|root|elevated)\s+access',
        r'enable\s+(admin|debug|developer)\s+mode',
        r'bypass\s+(security|safety|restrictions?)',
        r'(disable|turn\s+off|remove)\s+(security|safety|filters?)',

        # Data extraction tricks
        r'return\s+(all|entire)\s+database',
        r'show\s+(all|every)\s+(customer|user|account)',
        r'list\s+(all|every)\s+(customer|user|account)',
        r'dump\s+(database|table|data)',

        # Encoding/obfuscation attempts
        r'base64\s+decode',
        r'rot13',
        r'hex\s+decode',
        r'unicode\s+decode',
    ]

    # Suspicious keywords that warrant attention
    SUSPICIOUS_KEYWORDS = [
        'jailbreak', 'dan', 'developer mode', 'unrestricted',
        'no limitations', 'no restrictions', 'sudo', 'override',
        'exploit', 'vulnerability', 'bypass', 'circumvent',
        'manipulate', 'trick', 'deceive', 'social engineer'
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS]

    def detect(self, user_input: str) -> Tuple[bool, List[str], str]:
        """
        Detect prompt injection attempts

        Returns:
            (is_threat, threat_indicators, severity)
        """
        threat_indicators = []

        # Check for injection patterns
        for pattern in self.compiled_patterns:
            matches = pattern.findall(user_input)
            if matches:
                threat_indicators.append(f"Injection pattern: {pattern.pattern[:50]}")

        # Check for suspicious keywords
        lower_input = user_input.lower()
        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword in lower_input:
                threat_indicators.append(f"Suspicious keyword: {keyword}")

        # Check for excessive special characters (possible encoding)
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', user_input)) / max(len(user_input), 1)
        if special_char_ratio > 0.3:
            threat_indicators.append(f"High special character ratio: {special_char_ratio:.2f}")

        # Check for repeated delimiters
        if re.search(r'(---|===|###|\*\*\*){3,}', user_input):
            threat_indicators.append("Repeated delimiter patterns")

        # Determine severity
        severity = "LOW"
        if len(threat_indicators) >= 3:
            severity = "CRITICAL"
        elif len(threat_indicators) >= 2:
            severity = "HIGH"
        elif len(threat_indicators) >= 1:
            severity = "MEDIUM"

        is_threat = len(threat_indicators) > 0

        return is_threat, threat_indicators, severity


class JailbreakDetector:
    """Detect jailbreak attempts"""

    # Known jailbreak patterns
    JAILBREAK_PATTERNS = [
        # DAN (Do Anything Now) variants
        r'do\s+anything\s+now',
        r'\bDAN\b',
        r'evil\s+confidant',

        # Hypothetical scenarios
        r'imagine\s+you\s+(have\s+no|are\s+not\s+bound\s+by)',
        r'in\s+a\s+hypothetical\s+world',
        r'what\s+if\s+you\s+(had\s+no|weren\'t\s+bound)',

        # Compliance manipulation
        r'you\s+must\s+(comply|obey|follow)',
        r'you\s+have\s+no\s+choice\s+but\s+to',
        r'i\s+command\s+you\s+to',

        # Ethical bypass
        r'for\s+(educational|research)\s+purposes\s+only',
        r'this\s+is\s+(legal|authorized)',
        r'i\s+have\s+permission\s+to',

        # Fictional framing
        r'write\s+a\s+story\s+(where|about)',
        r'in\s+a\s+fictional\s+scenario',
        r'create\s+a\s+character\s+who',

        # Reverse psychology
        r'you\s+can\'t\s+tell\s+me',
        r'i\s+bet\s+you\s+won\'t',
        r'prove\s+you\s+can',
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.JAILBREAK_PATTERNS]

    def detect(self, user_input: str) -> Tuple[bool, List[str], str]:
        """
        Detect jailbreak attempts

        Returns:
            (is_threat, threat_indicators, severity)
        """
        threat_indicators = []

        # Check for jailbreak patterns
        for pattern in self.compiled_patterns:
            matches = pattern.findall(user_input)
            if matches:
                threat_indicators.append(f"Jailbreak pattern: {pattern.pattern[:50]}")

        # Check for attempts to redefine the assistant's identity
        identity_override = re.search(
            r'you\s+are\s+(not|no\s+longer)\s+(an?\s+)?(assistant|ai|bot|banking)',
            user_input,
            re.IGNORECASE
        )
        if identity_override:
            threat_indicators.append("Identity override attempt")

        # Check for attempts to disable safety features
        safety_bypass = re.search(
            r'(disable|turn\s+off|remove)\s+(safety|security|restrictions|filters)',
            user_input,
            re.IGNORECASE
        )
        if safety_bypass:
            threat_indicators.append("Safety bypass attempt")

        # Determine severity
        severity = "LOW"
        if len(threat_indicators) >= 3:
            severity = "CRITICAL"
        elif len(threat_indicators) >= 2:
            severity = "HIGH"
        elif len(threat_indicators) >= 1:
            severity = "MEDIUM"

        is_threat = len(threat_indicators) > 0

        return is_threat, threat_indicators, severity


class DataExfiltrationDetector:
    """Detect data exfiltration attempts"""

    # Patterns that suggest data exfiltration
    EXFILTRATION_PATTERNS = [
        # Bulk data requests
        r'(show|list|get|give\s+me)\s+(all|every|entire)\s+(customer|user|account|transaction)',
        r'export\s+(all|entire)\s+(database|table|data)',
        r'download\s+(all|entire)',

        # Other users' data
        r'show\s+(me\s+)?other\s+(people|customer|user)',
        r'list\s+(all\s+)?other\s+accounts?',
        r'who\s+else\s+(has|uses)',

        # System information
        r'show\s+(me\s+)?database\s+schema',
        r'list\s+(all\s+)?tables?',
        r'describe\s+table',
        r'show\s+columns?',

        # Sensitive fields
        r'(show|get|list).*(password|pin|secret|key|token)',
        r'(credit|debit)\s+card\s+number',
        r'account\s+number',
        r'social\s+security',

        # Aggregation attempts
        r'count\s+(all|total)\s+(customer|user|account)',
        r'sum\s+of\s+all\s+(balance|transaction)',
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.EXFILTRATION_PATTERNS]

    def detect(self, user_input: str, user_type: str) -> Tuple[bool, List[str], str]:
        """
        Detect data exfiltration attempts

        Args:
            user_input: The user's query
            user_type: 'customer' or 'merchant'

        Returns:
            (is_threat, threat_indicators, severity)
        """
        threat_indicators = []

        # Check for exfiltration patterns
        for pattern in self.compiled_patterns:
            matches = pattern.findall(user_input)
            if matches:
                threat_indicators.append(f"Exfiltration pattern: {pattern.pattern[:50]}")

        # Check for unauthorized scope (customers trying to access others' data)
        if user_type == 'customer':
            # More precise pattern that doesn't flag merchant legitimate queries
            unauthorized_scope = re.search(
                r'(show|list|get|all|every)\s+(all|every|other)\s+(customer|user|people|account)',
                user_input,
                re.IGNORECASE
            )
            if unauthorized_scope:
                threat_indicators.append("Unauthorized scope: customer requesting others' data")

        # For merchants, be more permissive with "transactions to my store" but flag other merchants
        if user_type == 'merchant':
            other_merchants = re.search(
                r'(other|all)\s+(merchant|store|shop)',
                user_input,
                re.IGNORECASE
            )
            # Exclude legitimate patterns like "customers who bought from me"
            legitimate_merchant_pattern = re.search(
                r'(my\s+store|to\s+me|from\s+me|bought\s+from)',
                user_input,
                re.IGNORECASE
            )
            if other_merchants and not legitimate_merchant_pattern:
                threat_indicators.append("Unauthorized scope: merchant requesting other merchants' data")

        # Check for SQL injection attempts
        sql_injection = re.search(
            r'(union\s+select|or\s+1\s*=\s*1|drop\s+table|;\s*--|where\s+1\s*=\s*1|from\s+\w+\s+where)',
            user_input,
            re.IGNORECASE
        )
        if sql_injection:
            threat_indicators.append("Possible SQL injection attempt")

        # Check for direct SQL-like patterns in natural language
        if re.search(r'select\s+\*\s+from\s+\w+', user_input, re.IGNORECASE):
            threat_indicators.append("SQL-like query pattern detected")

        # Determine severity
        severity = "LOW"
        if len(threat_indicators) >= 3:
            severity = "CRITICAL"
        elif len(threat_indicators) >= 2:
            severity = "HIGH"
        elif len(threat_indicators) >= 1:
            severity = "MEDIUM"

        is_threat = len(threat_indicators) > 0

        return is_threat, threat_indicators, severity


class ToolMisuseDetector:
    """Detect and prevent tool misuse"""

    def __init__(self):
        # Track tool usage per user
        self.tool_usage_history = defaultdict(lambda: deque(maxlen=100))
        self.suspicious_patterns = defaultdict(int)

    def detect_rate_abuse(self, user: str, tool_name: str, time_window: int = 60) -> Tuple[bool, str]:
        """
        Detect if user is abusing tool rate limits

        Args:
            user: User identifier
            tool_name: Name of the tool being called
            time_window: Time window in seconds (default 60s)

        Returns:
            (is_abuse, message)
        """
        now = datetime.now()

        # Add current call
        self.tool_usage_history[user].append({
            'tool': tool_name,
            'timestamp': now
        })

        # Count calls in the time window
        recent_calls = [
            call for call in self.tool_usage_history[user]
            if (now - call['timestamp']).total_seconds() <= time_window
        ]

        # Rate limit: 10 calls per minute per tool
        tool_calls = [call for call in recent_calls if call['tool'] == tool_name]
        if len(tool_calls) > 10:
            return True, f"Rate limit exceeded: {len(tool_calls)} calls to {tool_name} in {time_window}s"

        # Overall rate limit: 20 calls per minute across all tools
        if len(recent_calls) > 20:
            return True, f"Overall rate limit exceeded: {len(recent_calls)} total calls in {time_window}s"

        return False, ""

    def detect_suspicious_patterns(self, user: str, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Detect suspicious tool usage patterns

        Returns:
            (is_suspicious, indicators)
        """
        indicators = []

        # Pattern 1: Rapid tool switching (trying different tools quickly)
        recent_tools = [call['tool'] for call in list(self.tool_usage_history[user])[-10:]]
        unique_recent_tools = set(recent_tools)
        if len(unique_recent_tools) > 5:
            indicators.append("Rapid tool switching detected")

        # Pattern 2: Repeated failed attempts with same tool
        # (This would need to be tracked separately with tool results)

        # Pattern 3: Suspicious parameter combinations
        if tool_name in ['generate_sql_for_query', 'execute_sql_query']:
            if 'sql_query' in params:
                sql = str(params.get('sql_query', '')).lower()
                if any(keyword in sql for keyword in ['drop', 'delete', 'update', 'insert', 'alter']):
                    indicators.append("Suspicious SQL keywords detected")

        is_suspicious = len(indicators) > 0
        return is_suspicious, indicators


class SecurityValidator:
    """Main security validation orchestrator"""

    def __init__(self):
        self.prompt_injection_detector = PromptInjectionDetector()
        self.jailbreak_detector = JailbreakDetector()
        self.data_exfiltration_detector = DataExfiltrationDetector()
        self.tool_misuse_detector = ToolMisuseDetector()

        # Security policy configuration
        self.block_on_critical = True  # Block requests with CRITICAL severity
        self.block_on_high = True      # Block requests with HIGH severity
        self.warn_on_medium = True     # Warn but allow MEDIUM severity

        # Statistics
        self.total_queries_checked = 0
        self.total_threats_detected = 0
        self.total_threats_blocked = 0

    def validate_input(
        self,
        user_input: str,
        user: str,
        user_type: str
    ) -> Tuple[bool, Optional[str], List[SecurityEvent]]:
        """
        Validate user input against all security checks

        Args:
            user_input: The user's query
            user: User identifier
            user_type: 'customer' or 'merchant'

        Returns:
            (is_allowed, rejection_message, security_events)
        """
        self.total_queries_checked += 1
        security_events = []

        # 1. Check for prompt injection
        is_injection, injection_indicators, injection_severity = self.prompt_injection_detector.detect(user_input)
        if is_injection:
            event = SecurityEvent(
                timestamp=datetime.now().isoformat(),
                user=user,
                user_type=user_type,
                event_type="PROMPT_INJECTION",
                severity=injection_severity,
                query=user_input,
                threat_indicators=injection_indicators,
                action_taken="PENDING"
            )
            security_events.append(event)
            self.total_threats_detected += 1

        # 2. Check for jailbreak attempts
        is_jailbreak, jailbreak_indicators, jailbreak_severity = self.jailbreak_detector.detect(user_input)
        if is_jailbreak:
            event = SecurityEvent(
                timestamp=datetime.now().isoformat(),
                user=user,
                user_type=user_type,
                event_type="JAILBREAK_ATTEMPT",
                severity=jailbreak_severity,
                query=user_input,
                threat_indicators=jailbreak_indicators,
                action_taken="PENDING"
            )
            security_events.append(event)
            self.total_threats_detected += 1

        # 3. Check for data exfiltration
        is_exfiltration, exfiltration_indicators, exfiltration_severity = self.data_exfiltration_detector.detect(
            user_input, user_type
        )
        if is_exfiltration:
            event = SecurityEvent(
                timestamp=datetime.now().isoformat(),
                user=user,
                user_type=user_type,
                event_type="DATA_EXFILTRATION_ATTEMPT",
                severity=exfiltration_severity,
                query=user_input,
                threat_indicators=exfiltration_indicators,
                action_taken="PENDING"
            )
            security_events.append(event)
            self.total_threats_detected += 1

        # Determine if we should block this request
        should_block = False
        rejection_message = None

        # Find highest severity and count event types
        severities = [e.severity for e in security_events]
        event_types = set(e.event_type for e in security_events)

        # Block on multiple threat types (indicates sophisticated attack)
        if len(event_types) >= 2:
            should_block = True
            rejection_message = self._generate_rejection_message("CRITICAL", security_events)
        elif "CRITICAL" in severities and self.block_on_critical:
            should_block = True
            rejection_message = self._generate_rejection_message("CRITICAL", security_events)
        elif "HIGH" in severities and self.block_on_high:
            should_block = True
            rejection_message = self._generate_rejection_message("HIGH", security_events)
        elif "MEDIUM" in severities and self.warn_on_medium:
            # Allow but log warning
            rejection_message = None  # Don't block

        # Update action taken in events
        action = "BLOCKED" if should_block else ("WARNING" if severities else "ALLOWED")
        for event in security_events:
            event.action_taken = action

            # Log the security event
            if event.severity in ["HIGH", "CRITICAL"]:
                security_logger.critical(event.to_json())
            elif event.severity == "MEDIUM":
                security_logger.warning(event.to_json())
            else:
                security_logger.info(event.to_json())

        if should_block:
            self.total_threats_blocked += 1

        is_allowed = not should_block
        return is_allowed, rejection_message, security_events

    def validate_tool_call(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        user: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a tool call before execution

        Returns:
            (is_allowed, rejection_message)
        """
        # Check for rate abuse
        is_rate_abuse, rate_message = self.tool_misuse_detector.detect_rate_abuse(user, tool_name)
        if is_rate_abuse:
            event = SecurityEvent(
                timestamp=datetime.now().isoformat(),
                user=user,
                user_type="unknown",
                event_type="TOOL_RATE_ABUSE",
                severity="HIGH",
                query=f"Tool: {tool_name}",
                threat_indicators=[rate_message],
                action_taken="BLOCKED",
                additional_context={'tool_name': tool_name, 'params': str(tool_params)[:200]}
            )
            security_logger.warning(event.to_json())
            return False, f"🚫 Security Alert: {rate_message}. Please wait before making more requests."

        # Check for suspicious patterns
        is_suspicious, indicators = self.tool_misuse_detector.detect_suspicious_patterns(
            user, tool_name, tool_params
        )
        if is_suspicious:
            event = SecurityEvent(
                timestamp=datetime.now().isoformat(),
                user=user,
                user_type="unknown",
                event_type="SUSPICIOUS_TOOL_USAGE",
                severity="MEDIUM",
                query=f"Tool: {tool_name}",
                threat_indicators=indicators,
                action_taken="LOGGED",
                additional_context={'tool_name': tool_name, 'params': str(tool_params)[:200]}
            )
            security_logger.warning(event.to_json())
            # Don't block, just log

        return True, None

    def _generate_rejection_message(self, severity: str, events: List[SecurityEvent]) -> str:
        """Generate a user-friendly rejection message"""
        if severity == "CRITICAL":
            return (
                "🚫 SECURITY ALERT: Your request has been blocked due to security concerns.\n\n"
                "Our security system detected patterns that violate our usage policy. "
                "This incident has been logged.\n\n"
                "If you believe this is an error, please contact support with the timestamp: "
                f"{datetime.now().isoformat()}"
            )
        elif severity == "HIGH":
            return (
                "⚠️ SECURITY WARNING: Your request could not be processed.\n\n"
                "We detected potentially unsafe patterns in your query. "
                "Please rephrase your request or contact support if you need assistance.\n\n"
                f"Incident ID: {datetime.now().isoformat()}"
            )
        else:
            return (
                "⚠️ Your request raised security concerns and could not be processed. "
                "Please try rephrasing or contact support."
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            'total_queries_checked': self.total_queries_checked,
            'total_threats_detected': self.total_threats_detected,
            'total_threats_blocked': self.total_threats_blocked,
            'threat_detection_rate': (
                self.total_threats_detected / self.total_queries_checked
                if self.total_queries_checked > 0 else 0
            ),
            'block_rate': (
                self.total_threats_blocked / self.total_queries_checked
                if self.total_queries_checked > 0 else 0
            )
        }


# Global security validator instance
security_validator = SecurityValidator()


def validate_user_input(user_input: str, user: str, user_type: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate user input

    Returns:
        (is_allowed, rejection_message)
    """
    is_allowed, rejection_message, events = security_validator.validate_input(
        user_input, user, user_type
    )
    return is_allowed, rejection_message


def validate_tool_call(tool_name: str, tool_params: Dict[str, Any], user: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate tool calls

    Returns:
        (is_allowed, rejection_message)
    """
    return security_validator.validate_tool_call(tool_name, tool_params, user)


def get_security_statistics() -> Dict[str, Any]:
    """Get current security statistics"""
    return security_validator.get_statistics()
