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

def test_google_search_returns_five_best(monkeypatch):
    def fake(query, k):
        return [{"title": f"T{i}", "href": f"http://e/{i}", "body": f"snippet {i}"}
                for i in range(10)]
    monkeypatch.setattr(websearch, "_backend", fake)
    out = websearch.google_search("who invented radio", k=5)
    assert out.count("http://e/") == 5           # capped at 5
    assert "T0" in out and "snippet 0" in out


def test_google_search_edge_cases(monkeypatch):
    assert websearch.google_search("") == "No query given."
    monkeypatch.setattr(websearch, "_backend", lambda q, k: [])
    assert websearch.google_search("nothing") == "No results found."
    monkeypatch.setattr(websearch, "_backend",
                        lambda q, k: (_ for _ in ()).throw(RuntimeError("net down")))
    assert "Web search failed" in websearch.google_search("boom")


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
