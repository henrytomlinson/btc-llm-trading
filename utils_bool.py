"""
Boolean parsing utilities for the Bitcoin LLM Trading System
Handles robust conversion of various string/numeric values to boolean
"""

def parse_bool(v, default=False):
    """
    Robustly parse a value to boolean.
    
    Args:
        v: Value to parse (string, bool, int, None, etc.)
        default: Default value if parsing fails
        
    Returns:
        bool: Parsed boolean value
        
    Examples:
        parse_bool("True") -> True
        parse_bool("False") -> False
        parse_bool("1") -> True
        parse_bool("0") -> False
        parse_bool("yes") -> True
        parse_bool("no") -> False
        parse_bool("") -> False
        parse_bool(None) -> False
        parse_bool(True) -> True
        parse_bool(False) -> False
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    
    # Convert to string and normalize
    s = str(v).strip().lower()
    
    # Handle empty strings
    if not s:
        return default
    
    # True values
    if s in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}:
        return True
    
    # False values
    if s in {"0", "false", "f", "no", "n", "off", "disabled", "disable"}:
        return False
    
    # If we can't parse it, return default
    return default
