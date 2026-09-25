"""Standalone graph-based APLS and TLTS metrics."""

__all__ = ["APLS", "TLTS"]


def __getattr__(name: str):
    if name == "APLS":
        from .apls import APLS

        return APLS
    if name == "TLTS":
        from .tlts import TLTS

        return TLTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
