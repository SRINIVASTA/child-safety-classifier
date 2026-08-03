import hashlib
import re
import os

LOG_SALT = os.getenv("THORN_LOG_SALT", "default_secure_salt_string_12345").encode('utf-8')

def anonymize_text(text: str) -> str:
    """Scrambles personal text identifying data via SHA-256 hashing for privacy compliance."""
    if not text:
        return ""
    sanitized = text.strip()
    sanitized = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', sanitized)
    sanitized = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP_REDACTED]', sanitized)
    
    hasher = hashlib.sha256()
    hasher.update(LOG_SALT)
    hasher.update(sanitized.encode('utf-8'))
    return f"SHA256_{hasher.hexdigest()[:32]}"
