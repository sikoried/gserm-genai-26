"""Connectivity helpers for the online tools.

When the machine has no internet, the online tools must fail gracefully — return a
clear message rather than raise — so the agent still terminates and the synthesizer
can tell the user the lookup could not be performed (see tools.md, "Offline
resilience").
"""
from __future__ import annotations

import socket

# Substrings that mark a connectivity failure across the various HTTP stacks
# (httpx/primp for ddgs, yt-dlp, youtube-transcript-api).
_OFFLINE_MARKERS = (
    "getaddrinfo", "nodename nor servname", "name or service not known",
    "temporary failure in name resolution", "failed to resolve", "no address associated",
    "connection refused", "network is unreachable", "connection aborted",
    "connection error", "max retries", "timed out", "timeout", "unreachable",
    "no internet", "name resolution",
)


def is_offline_error(exc: BaseException) -> bool:
    """True if `exc` looks like a lack of internet connectivity."""
    if isinstance(exc, (socket.gaierror, TimeoutError, ConnectionError)):
        return True
    return any(m in str(exc).lower() for m in _OFFLINE_MARKERS)


def offline_message(what: str) -> str:
    return f"No internet connection — {what} unavailable."
