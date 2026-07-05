"""Network-free tests for the online tools — the HTTP backends are mocked.

``google_search`` (DuckDuckGo) and ``youtube`` (yt-dlp + transcript API) reach the
network in production; here we monkeypatch their ``_backend`` / ``_search_backend``
/ ``_transcript_backend`` functions so nothing leaves the process.
"""
import importlib

# The package binds the smolagents-wrapped Tool under the same name, so reach the
# real submodules via importlib to monkeypatch their backends.
websearch = importlib.import_module("oracle.tools.websearch")
youtube = importlib.import_module("oracle.tools.youtube")


# --- google_search ------------------------------------------------------------

def test_google_search_returns_up_to_ten_best(monkeypatch):
    def fake(query, k):
        return [{"title": f"T{i}", "href": f"http://e/{i}", "body": f"snippet {i}"}
                for i in range(15)]
    monkeypatch.setattr(websearch, "_backend", fake)
    # Default is now 10 results.
    out = websearch.google_search("who invented radio")
    assert out.count("http://e/") == 10
    assert "T0" in out and "snippet 0" in out
    # An explicit smaller k is still honoured; k above 10 is capped at 10.
    assert websearch.google_search("q", k=3).count("http://e/") == 3
    assert websearch.google_search("q", k=50).count("http://e/") == 10


def test_google_search_edge_cases(monkeypatch):
    assert websearch.google_search("") == "No query given."
    monkeypatch.setattr(websearch, "_backend", lambda q, k: [])
    assert websearch.google_search("nothing") == "No results found."
    monkeypatch.setattr(websearch, "_backend",
                        lambda q, k: (_ for _ in ()).throw(RuntimeError("parse error")))
    assert "Web search failed" in websearch.google_search("boom")


def test_google_search_offline_is_graceful(monkeypatch):
    # No internet must not crash — return a clear "no internet" observation.
    def offline(q, k):
        raise ConnectionError("Failed to resolve 'duckduckgo.com' (getaddrinfo failed)")
    monkeypatch.setattr(websearch, "_backend", offline)
    out = websearch.google_search("who is the current CEO of OpenAI")
    assert "No internet connection" in out and "web search unavailable" in out


def test_youtube_offline_is_graceful(monkeypatch):
    def offline(q, k):
        raise TimeoutError("connection timed out")
    monkeypatch.setattr(youtube, "_search_backend", offline)
    out = youtube.youtube("mark rober mousetrap")
    assert "No internet connection" in out and "YouTube search unavailable" in out


# --- youtube ------------------------------------------------------------------

def test_youtube_extracts_transcript(monkeypatch):
    monkeypatch.setattr(youtube, "_search_backend",
                        lambda q, k: [{"id": "v1", "title": "Vid", "description": "desc"}])
    monkeypatch.setattr(youtube, "_transcript_backend", lambda vid: "the answer is 42")
    out = youtube.youtube("some quiz topic", k=5)
    assert "Vid" in out and "the answer is 42" in out and "transcript" in out


def test_youtube_falls_back_to_description(monkeypatch):
    monkeypatch.setattr(youtube, "_search_backend",
                        lambda q, k: [{"id": "v1", "title": "Vid", "description": "desc fallback"}])
    monkeypatch.setattr(youtube, "_transcript_backend",
                        lambda vid: (_ for _ in ()).throw(Exception("captions disabled")))
    out = youtube.youtube("topic")
    assert "desc fallback" in out and "no transcript" in out


def test_youtube_edge_cases(monkeypatch):
    assert youtube.youtube("") == "No query given."
    monkeypatch.setattr(youtube, "_search_backend", lambda q, k: [])
    assert youtube.youtube("nothing") == "No videos found."
