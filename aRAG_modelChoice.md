# Agentic RAG — Split-Model Architecture

## Overview

The agentic RAG system uses a **two-model split** to balance cost, latency, and answer quality. A small, locally hosted LLM handles the routing decision — whether the question requires retrieval from the knowledge base or can be answered from world knowledge alone — while a large cloud-based LLM generates the final answer.

## Motivation

The current agentic RAG implementation routes every step through a single large cloud model (e.g. Mistral Medium 128B). This means the reasoning about *whether* to retrieve, *what* to search for, and *how* to rephrase a query all consume expensive cloud tokens and add network latency. In practice, the routing decision ("does this question need external context?") is a straightforward classification task that does not require the full capability of a large model.

By offloading the routing decision to a small local model, we achieve:

- **Lower cost**: the routing step consumes zero cloud tokens.
- **Lower latency**: local inference avoids a network round-trip for the decision step.
- **Privacy**: the user's question is only sent to the cloud once the routing decision has been made, and only for answer generation — the decision itself stays on-device.

## Architecture

```
User question
      │
      ▼
┌─────────────┐
│  Local LLM  │  (small, e.g. Qwen 3B / Phi-3 Mini / Mistral Small)
│  "Router"   │
└──────┬──────┘
       │
       ├── needs_retrieval = false ──► skip retrieval
       │
       ├── needs_retrieval = true  ──► embed query → FAISS search → context
       │                               (optionally: local LLM rewrites query first)
       │
       ▼
┌─────────────┐
│  Cloud LLM  │  (large, e.g. Mistral Medium 128B / GPT-OSS 120B)
│  "Answerer"  │
└──────┬──────┘
       │
       ▼
   Final answer
```

### Step 1 — Routing (local small LLM)

The local model receives the user's question (and conversation history, if any) and classifies it into one of two categories:

- **world**: the question can be answered from general knowledge without retrieval.
- **rag**: the question requires specific information that should be retrieved from the wiki-10k knowledge base.

The local model may also produce a rewritten search query when it decides retrieval is needed, so the embedding better captures the intent of the question.

The prompt for the router should be structured to return a short, parseable output (e.g. JSON) to keep generation fast:

```
Classify whether the following question needs retrieval from a knowledge base
or can be answered from general world knowledge.

Question: {question}

Respond with JSON only: {"route": "world" | "rag", "search_query": "...or null"}
```

### Step 2 — Retrieval (conditional)

If the router returns `rag`, the search query (either the original question or the rewritten version from Step 1) is embedded locally using sentence-transformers and searched against the FAISS index. The top-k chunks are assembled into context, identical to the existing RAG pipeline.

If the router returns `world`, this step is skipped entirely.

### Step 3 — Answer generation (cloud large LLM)

The large cloud model receives:

- The system prompt (with or without retrieved context, depending on the route).
- The conversation history.
- The user's question.

It generates the final answer. This is the only step that incurs cloud API cost.

## Configuration

The split-model approach requires two model references in the QA config:

- **router_model**: the local small LLM used for the routing decision (runs locally via a local inference server, e.g. Ollama or vLLM).
- **model**: the cloud LLM used for answer generation (unchanged from the current config, routed through the proxy).

A new `router_endpoint` field points to the local inference server, keeping it separate from the cloud proxy endpoint.

```yaml
type: a-rag
model: mistralai/Mistral-Medium-3.5-128B        # cloud answerer
endpoint: https://kiz1.in.ohmportal.de/llmproxy/v1
router_model: qwen2.5:3b                         # local router
router_endpoint: http://localhost:11434/v1        # local Ollama
temperature: 0.0
```

## Trade-offs

| Aspect | Benefit | Risk |
|--------|---------|------|
| Cost | Routing step is free (local) | Misrouted questions may produce worse answers |
| Latency | No network round-trip for routing | Local model inference adds some local compute time |
| Quality | Large model focuses on what it does best (generation) | Small model may misclassify edge cases (e.g. questions that seem general but need specific wiki context) |
| Complexity | Clear separation of concerns | Two models to manage; local inference server must be running |

## Fallback Behavior

If the local router is unavailable (server down, model not loaded), the system should fall back to always routing through RAG — the safer default, since retrieving unnecessary context is less harmful than missing needed context.
