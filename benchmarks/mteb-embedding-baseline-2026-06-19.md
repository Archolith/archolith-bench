# MTEB embedding-model baseline — 2026-06-19

<!-- archolith-evidence
product: menhir
command: python scripts/run_mteb_local.py
commit: unknown
run_date: 2026-06-19
source: MTEB 2.15.5, SciFact retrieval dataset
source_tracked: false
public_copy_allowed: false
note: Single-arm embedding baseline. The proxy is not in the embeddings path.
-->

**Type:** Single-arm embedding-model baseline (NOT a proxy A/B). MTEB measures embedding
quality; the Archolith chat proxy is not in the embeddings path. This number describes the
embedding model that menhir / fact-retrieval depends on, not the proxy.

**Harness:** Official MTEB 2.15.5 + `datasets`, real SciFact retrieval dataset.
**Runner:** `scripts/run_mteb_local.py` (in-process `AbsEncoder` subclass calling an
OpenAI-compatible `/embeddings` endpoint).

Both numbers below are **measured** here (same harness, same SciFact dataset):

| Model | Endpoint | Metric | Score | Cost |
|-------|----------|--------|-------|------|
| `text-embedding-3-small` (menhir's ACTIVE embedder) | OpenAI `api.openai.com/v1` | NDCG@10 | **0.72964** | paid |
| `text-embedding-nomic-embed-text-v1.5` | LM Studio `localhost:1234/v1` | NDCG@10 | **0.68115** | free (local) |
| **Gap (paid − free)** | | | **0.04849** (~6.6% relative) | |

## Decision context (menhir embedding choice)

- menhir's **active** embedder is OpenAI `text-embedding-3-small` (1536-dim), per
  `GRAPHITI_EMBED_PROVIDER=openai` / `OPENAI_EMBED_MODEL`. Measured SciFact = 0.72964, matching
  the published ~0.73 (validates the runner).
- The free local nomic (LM Studio) measures 0.68115 — **~4.85 NDCG@10 points (~6.6% relative)
  behind** on SciFact retrieval. That is the cost, in retrieval quality, of dropping menhir to
  free local embeddings.
- menhir also has a *configured but inactive* local embed path at `localhost:8083/v1`
  (llama.cpp `nomic-embed-text-v1.5.Q4_K_M.gguf`) — a different endpoint and quantization (Q4_K_M)
  than the LM Studio `:1234` server benchmarked here, so its score could differ (quantization loss).

## Caveats

- One task (SciFact), one metric. Not a full MTEB suite run; a switch decision should sample more
  retrieval tasks (e.g. NFCorpus, FiQA, TREC-COVID).
- The OpenAI run initially hit a transient 429; re-run succeeded with throttling + Retry-After
  backoff (batch 32, 0.15s spacing).
- These are embedding-model baselines, not advertisable "Archolith scores." A proxy A/B requires
  an embeddings proxy/caching layer that does not exist.

## Reproduce

```bash
pip install mteb
python scripts/run_mteb_local.py SciFact
# EMBEDDINGS_BASE_URL / EMBEDDINGS_MODEL override the endpoint/model.
```
