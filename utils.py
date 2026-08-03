import hashlib
import re
import os

# Production-grade salt rotation. Defaults to static fallback if env var is missing.
LOG_SALT = os.getenv("THORN_LOG_SALT", "default_secure_salt_string_12345").encode('utf-8')

def anonymize_text(text: str) -> str:
    """
    Scrambles explicit or personal text identifying data via SHA-256 hashing.
    Also strips out typical email and IP configurations before recording logs.
    """
    if not text:
        return ""
    
    # 1. Clean common PII variables via Regex
    sanitized = text.strip()
    sanitized = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', sanitized)
    sanitized = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP_REDACTED]', sanitized)
    
    # 2. Apply cryptographic hashing to the remaining text block
    hasher = hashlib.sha256()
    hasher.update(LOG_SALT)
    hasher.update(sanitized.encode('utf-8'))
    
    # Return truncated hex digest to keep server log strings compact
    return f"SHA256_{hasher.hexdigest()[:32]}"
