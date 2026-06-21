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


def record(event: dict) -> int:
    """Record a telemetry event and return the call row id."""
    event["ts"] = time.time()
    logger.info(json.dumps(event))
    return log_call(event)


def observed(tool_name: str):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                text, meta = await fn(*args, **kwargs)
                input_text = " ".join(
                    str(v) for v in list(args) + list(kwargs.values())
                )
                call_id = record(
                    {
                        "tool": tool_name,
                        "ok": True,
                        "input_chars": len(input_text),
                        "output_chars": len(text),
                        "total_ms": int((time.perf_counter() - t0) * 1000),
                        **meta,
                    }
                )

                from .grading import schedule_grading
                from .grading.heuristics import run_heuristics

                schedule_grading(tool_name, input_text, text, call_id)

                failures = [
                    r for r in run_heuristics(tool_name, input_text, text)
                    if not r["passed"]
                ]
                if failures:
                    warnings = "\n".join(
                        f"  - {r['checker']}: {r['details'].get('reason', 'failed')}"
                        for r in failures
                    )
                    text += (
                        f"\n\n"
                        f"IMPORTANT — Quality warning: the model output failed "
                        f"{len(failures)} automated quality check{'s' if len(failures) != 1 else ''}. "
                        f"Please inform the user about these issues:\n"
                        f"{warnings}\n"
                        f"The user should be aware that this output may need corrections."
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
