"""YouTube tool (online, opt-in): find videos and extract answer-bearing text.

Cost-free and key-free: video search uses ``yt-dlp`` (``ytsearchN:``, metadata
only — nothing is downloaded), and transcripts use ``youtube-transcript-api``. The
tool returns a compact digest — title + transcript/description excerpt per video —
so the synthesizer can read the answer out of it, not just links.

Both network calls live in ``_search_backend`` / ``_transcript_backend`` so tests
can monkeypatch them and stay network-free. No LLM is called (0 tokens); a single
call can be token-heavy, so results and transcript length are capped.
"""
from __future__ import annotations

_MAX_TRANSCRIPT_CHARS = 800


def _search_backend(query: str, k: int) -> list[dict]:
    """Return up to `k` videos as {id, title, description} dicts (lazy import)."""
    import yt_dlp
    opts = {"quiet": True, "skip_download": True, "extract_flat": True,
            "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{k}:{query}", download=False) or {}
    out = []
    for entry in (info.get("entries") or [])[:k]:
        out.append({"id": entry.get("id", ""),
                    "title": entry.get("title", ""),
                    "description": entry.get("description", "") or ""})
    return out


def _transcript_backend(video_id: str) -> str:
    """Return the plain-text transcript for a video id, or "" (lazy import)."""
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)  # FetchedTranscript: iterable of snippets
    return " ".join(snippet.text for snippet in fetched)


def _digest_one(video: dict) -> str:
    vid = video.get("id", "")
    title = (video.get("title") or "").strip()
    text = ""
    try:
        text = _transcript_backend(vid) if vid else ""
    except Exception:
        text = ""  # captions disabled / unavailable → fall back to description
    source = "transcript"
    if not text:
        text = video.get("description") or ""
        source = "description (no transcript)"
    text = " ".join(text.split())[:_MAX_TRANSCRIPT_CHARS]
    return f"{title}\n[{source}] {text}" if text else f"{title}\n[no transcript or description]"


def youtube(query: str, k: int = 5) -> str:
    """Find the best matching YouTube videos and extract text to answer from.

    Searches YouTube, then for each of up to `k` videos (default 5) returns its
    transcript (falling back to the description) — a digest you can read the answer
    out of, not just links. Online tool; may be empty.

    Args:
        query: What to look for on YouTube.
        k: Maximum number of videos to inspect (capped at 5).
    """
    q = (query or "").strip()
    if not q:
        return "No query given."
    k = max(1, min(int(k or 5), 5))
    try:
        videos = _search_backend(q, k) or []
    except Exception as exc:
        return f"YouTube search failed: {exc}"
    if not videos:
        return "No videos found."
    return "\n\n".join(f"[{i}] {_digest_one(v)}" for i, v in enumerate(videos[:k], 1))
