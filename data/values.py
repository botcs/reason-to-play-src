"""Tagged JSON values for recorded metadata."""

import base64
from datetime import datetime
import math

_TAG = "$rtp"


class BinaryValue(bytes):
    """Bytes carrying the original binary subtype."""

    def __new__(cls, value: bytes, subtype: int = 0):
        obj = super().__new__(cls, value)
        obj.subtype = subtype
        return obj


def encode_value(value):
    """Convert source values to unambiguous JSON-compatible values."""
    if isinstance(value, bytes):
        return {
            _TAG: "binary",
            "base64": base64.b64encode(value).decode("ascii"),
            "subtype": getattr(value, "subtype", 0),
        }
    if isinstance(value, datetime):
        return {_TAG: "datetime", "iso": value.isoformat()}
    if isinstance(value, dict):
        if _TAG in value or any(not isinstance(key, str) for key in value):
            return {
                _TAG: "mapping",
                "items": [[encode_value(k), encode_value(v)] for k, v in value.items()],
            }
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {_TAG: "float", "value": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported source value type: {type(value).__name__}")


def decode_value(value):
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get(_TAG)
    if kind == "binary":
        return BinaryValue(
            base64.b64decode(value["base64"], validate=True), value["subtype"]
        )
    if kind == "datetime":
        return datetime.fromisoformat(value["iso"])
    if kind == "mapping":
        return {decode_value(k): decode_value(v) for k, v in value["items"]}
    if kind == "float":
        if value["value"] not in {"nan", "inf", "-inf"}:
            raise ValueError("Invalid tagged float")
        return float(value["value"])
    if kind is not None:
        raise ValueError(f"Unknown canonical value type: {kind}")
    return {key: decode_value(item) for key, item in value.items()}
