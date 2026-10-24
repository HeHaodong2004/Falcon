"""Optional lossless JSON decode acceleration; stdlib fallback is identical."""
import json
try:
    import orjson
except ImportError:
    orjson=None


def loads(data):return orjson.loads(data) if orjson is not None else json.loads(data)
