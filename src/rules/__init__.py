"""Rules package — importing this module auto-registers all rules."""

from src.rules.registry import ToolRegistry, registry

# Import rule modules to trigger registration
from src.rules import security  # noqa: F401
from src.rules import complexity  # noqa: F401
from src.rules import bug_risk  # noqa: F401
from src.rules import style  # noqa: F401
from src.rules import performance  # noqa: F401
from src.rules import maintainability  # noqa: F401

# llm_assisted is NOT auto-registered here because it needs an LLM client.
# It is registered dynamically in main.py's _init_components().

__all__ = ["ToolRegistry", "registry"]
