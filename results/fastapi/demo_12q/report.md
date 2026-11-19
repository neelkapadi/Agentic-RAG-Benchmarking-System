# Benchmark report: demo_12q

- **Corpus:** `fastapi` (the FastAPI web framework documentation)
- **Questions:** 12 per pipeline · top-k = 5
- **Generator / judge:** `qwen2.5:3b` / `llama3.2:3b` · embeddings `BAAI/bge-small-en-v1.5`
- **Generated:** 2026-11-20 03:22

## Key findings

- Best **Hit@5**: `hybrid` (1.000)
- Best **MRR**: `hybrid` (0.892)
- Best **faithfulness**: `bm25` (0.963)
- Best **correctness**: `agentic` (0.938)
- Agentic vs best baseline (`hybrid`) on Hit@5: -8.3 points
- Agentic vs best baseline (`vector`) on correctness: +6.2 points
- Agentic median latency is 4.5x `hybrid` (34.22s vs 7.68s)

## Quality (higher is better, except abstained)

| pipeline | hit@5 | recall@5 | mrr | ndcg@5 | doc_hit@5 | faithfulness | answer_relevance | correctness | abstained |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.917 | 0.917 | 0.854 | 0.869 | **1.000** | **0.963** | 0.917 | 0.854 | 0.083 |
| vector | 0.833 | 0.833 | 0.750 | 0.772 | **1.000** | 0.955 | 0.958 | 0.875 | 0.000 |
| hybrid | **1.000** | **1.000** | **0.892** | **0.918** | **1.000** | 0.948 | 0.917 | 0.854 | 0.083 |
| agentic | 0.917 | 0.917 | 0.736 | 0.783 | **1.000** | 0.959 | **0.979** | **0.938** | 0.000 |

## Latency and cost (lower is better)

| pipeline | retrieval_s_p50 | retrieval_s_p95 | total_s_p50 | total_s_p95 | llm_calls_mean | prompt_mean | completion_mean |
|---|---|---|---|---|---|---|---|
| bm25 | **0.00** | **0.00** | 9.69 | 12.04 | **1.00** | 1227.42 | 106.83 |
| vector | 0.02 | 0.04 | 8.57 | 12.04 | **1.00** | 1228.50 | **101.08** |
| hybrid | 0.02 | 0.02 | **7.68** | **10.34** | **1.00** | **1216.83** | 104.08 |
| agentic | 30.34 | 33.84 | 34.22 | 42.86 | 7.50 | 4026.08 | 310.42 |

## hit@5 by question type

| question type | bm25 | vector | hybrid | agentic |
|---|---|---|---|---|
| conceptual | 1.000 | 0.833 | 1.000 | 1.000 |
| keyword | 0.833 | 0.833 | 1.000 | 0.833 |

## Agent behaviour

- First strategy chosen: `vector` 7, `hybrid` 4, `bm25` 1
- Average retrieve→grade attempts: 2.50
- Context accepted by grader: 3/12 (25%); the rest fell back to fusing all attempts

Routing by question type:

| type | bm25 | vector | hybrid |
|---|---|---|---|
| conceptual | 0 | 5 | 1 |
| keyword | 1 | 2 | 3 |

Questions where the agent missed the gold chunk (up to 5):

- **q009** (keyword): How do I start my FastAPI app for development? — strategies vector

## Metric definitions

- **Hit@5 / Recall@5 / MRR / nDCG@5**: was the gold chunk retrieved, and how high.
- **Doc-Hit@5**: looser; any chunk from the gold document counts.
- **Faithfulness**: share of answer claims supported by the retrieved context (LLM judge).
- **Answer relevance / correctness**: LLM judge 1–5 scores rescaled to 0–1.
- **Abstained**: share of answers that said "I don't know".
