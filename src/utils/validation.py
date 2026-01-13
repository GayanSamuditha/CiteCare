"""
Shared validation utilities.
"""

import re
from typing import Tuple


def validate_collection_name(name: str) -> Tuple[bool, str]:
    """
    Validate collection name according to Chroma requirements.

    Rules:
    - 3 to 512 characters
    - Letters, numbers, dots, underscores, hyphens
    - Must start and end with alphanumeric
    """
    if not name:
        return False, "Name is required"
    if len(name) < 3:
        return False, "Name must be at least 3 characters"
    if len(name) > 512:
        return False, "Name must be less than 512 characters"
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$", name):
        return False, "Only letters, numbers, dots, underscores, and hyphens; start/end with alphanumeric"
    return True, ""
