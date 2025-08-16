from datetime import datetime, timezone

def utc_now() -> datetime:
    """Get current UTC datetime (replacement for deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc)
