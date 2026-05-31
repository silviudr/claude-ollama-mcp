"""Privacy intercept — detect sensitive content before processing."""

from __future__ import annotations

import fnmatch
import functools
import json
import re
from dataclasses import dataclass, field

from .config import PRIVACY_CONFIG_PATH
from .telemetry import record

DEFAULT_FILE_PATTERNS = [
    "*.env",
    "*.env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "credentials.json",
    "service-account*.json",
    "secrets/*",
    "customer_data/*",
    ".aws/*",
    ".ssh/*",
]

DEFAULT_CONTENT_PATTERNS = [
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
    r"AKIA[0-9A-Z]{16}",
    r"ghp_[A-Za-z0-9_]{36,}",
    r"sk-[A-Za-z0-9]{32,}",
    r"(?i)password\s*[=:]\s*\S+",
    r"(?i)api[_-]?key\s*[=:]\s*\S+",
    r"(?i)secret[_-]?key\s*[=:]\s*\S+",
]


@dataclass
class PrivacyConfig:
    file_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_FILE_PATTERNS))
    content_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_CONTENT_PATTERNS)
    )
    action: str = "warn"  # "warn", "redact_log", "reject"

    @classmethod
    def load(cls) -> PrivacyConfig:
        if not PRIVACY_CONFIG_PATH.exists():
            return cls()
        try:
            data = json.loads(PRIVACY_CONFIG_PATH.read_text())
            return cls(
                file_patterns=data.get("file_patterns", DEFAULT_FILE_PATTERNS),
                content_patterns=data.get("content_patterns", DEFAULT_CONTENT_PATTERNS),
                action=data.get("action", "warn"),
            )
        except (json.JSONDecodeError, OSError):
            return cls()


@dataclass
class PrivacyMatch:
    kind: str  # "file_pattern" or "content_pattern"
    pattern: str
    matched_text: str


def scan(text: str, config: PrivacyConfig | None = None) -> list[PrivacyMatch]:
    if config is None:
        config = PrivacyConfig.load()

    matches: list[PrivacyMatch] = []

    for pattern in config.file_patterns:
        for word in text.split():
            cleaned = word.strip("\"'`,;:()")
            if fnmatch.fnmatch(cleaned, pattern) or fnmatch.fnmatch(
                cleaned.split("/")[-1], pattern
            ):
                matches.append(PrivacyMatch("file_pattern", pattern, cleaned))

    compiled = []
    for pat in config.content_patterns:
        try:
            compiled.append((pat, re.compile(pat)))
        except re.error:
            continue

    for pat_str, regex in compiled:
        for m in regex.finditer(text):
            matched = m.group(0)
            redacted = matched[:6] + "***" if len(matched) > 6 else "***"
            matches.append(PrivacyMatch("content_pattern", pat_str, redacted))

    return matches


def privacy_guard(fn):
    """Decorator that scans tool inputs for sensitive content."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        config = PrivacyConfig.load()

        input_text = " ".join(str(v) for v in list(args) + list(kwargs.values()))
        matches = scan(input_text, config)

        if matches:
            descriptions = [f"{m.kind}: {m.pattern}" for m in matches]
            record({
                "event": "privacy_intercept",
                "tool": fn.__name__,
                "matches": descriptions,
                "action": config.action,
            })

            if config.action == "reject":
                matched_summary = ", ".join(descriptions)
                raise PrivacyError(
                    f"Rejected: input contains sensitive content matching "
                    f"[{matched_summary}]. Configure privacy rules in "
                    f"{PRIVACY_CONFIG_PATH} or set action to 'warn'."
                )

        return await fn(*args, **kwargs)

    return wrapper


class PrivacyError(Exception):
    """Raised when privacy policy rejects a request."""
