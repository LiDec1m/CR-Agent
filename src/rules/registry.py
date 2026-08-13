"""ToolRegistry: central registration and execution of deterministic rules."""

from __future__ import annotations

from typing import Callable

from src.models import Evidence, HunkInfo

RuleFunc = Callable[[HunkInfo], list[Evidence]]


class ToolRegistry:
    """Registry for deterministic code analysis rules."""

    def __init__(self) -> None:
        self._rules: dict[str, RuleFunc] = {}

    def register(self, name: str, func: RuleFunc) -> None:
        self._rules[name] = func

    def get(self, name: str) -> RuleFunc:
        return self._rules[name]

    def list_all(self) -> list[str]:
        return sorted(self._rules.keys())

    def execute(self, name: str, hunk: HunkInfo) -> list[Evidence]:
        func = self._rules.get(name)
        if func is None:
            return []
        return func(hunk)

    def execute_batch(
        self, names: list[str], hunks: list[HunkInfo]
    ) -> list[Evidence]:
        results: list[Evidence] = []
        for hunk in hunks:
            for name in names:
                results.extend(self.execute(name, hunk))
        return results


# Global singleton
registry = ToolRegistry()
