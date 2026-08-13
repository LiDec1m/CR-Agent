from .long_term import LongTermMemory


def __getattr__(name: str):
    if name == "ShortTermMemory":
        from .short_term import ShortTermMemory

        return ShortTermMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ShortTermMemory", "LongTermMemory"]
