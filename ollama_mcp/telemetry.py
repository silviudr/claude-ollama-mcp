"""JSON-lines logging, SQLite storage, and the observed() tool decorator."""

import functools
import json
import logging
import time

from .config import LOG_PATH
from .storage import log_call

logger = logging.getLogger("ollama_mcp")
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False


def record(event: dict) -> None:
    event["ts"] = time.time()
    logger.info(json.dumps(event))
    log_call(event)


def observed(tool_name: str):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                text, meta = await fn(*args, **kwargs)
                record(
                    {
                        "tool": tool_name,
                        "ok": True,
                        "input_chars": sum(
                            len(str(v)) for v in list(args) + list(kwargs.values())
                        ),
                        "output_chars": len(text),
                        "total_ms": int((time.perf_counter() - t0) * 1000),
                        **meta,
                    }
                )
                return text
            except Exception as e:
                record(
                    {
                        "tool": tool_name,
                        "ok": False,
                        "error": repr(e),
                        "total_ms": int((time.perf_counter() - t0) * 1000),
                    }
                )
                raise

        return wrapper

    return deco
