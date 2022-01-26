"""Get the metadata from setup.py."""
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

try:
    data = metadata.metadata("pyPCGA")
    __version__ = data["Version"]
    __author__ = data["Author"]
    __name__ = data["Name"]
except Exception:
    __version__ = "unknown"
    __author__ = "unknown"
    __name__ = "unknown"
