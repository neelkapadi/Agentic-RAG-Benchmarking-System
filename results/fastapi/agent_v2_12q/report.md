# Benchmark report: agent_v2_12q

- **Corpus:** `fastapi` (the FastAPI web framework documentation)
- **Questions:** 12 per pipeline · top-k = 5
- **Generator / judge:** `qwen2.5:3b` / `llama3.2:3b` · embeddings `BAAI/bge-small-en-v1.5`
- **Generated:** 2026-11-20 04:54

## Key findings

- Best **Hit@5**: `hybrid` (1.000)
- Best **MRR**: `hybrid` (0.892)
- Best **faithfulness**: `bm25` (0.963)
- Best **correctness**: `vector` (0.875)
- Agentic vs best baseline (`hybrid`) on Hit@5: -16.7 points
- Agentic vs best baseline (`vector`) on correctness: -6.2 points
- Agentic median latency is 1.9x `hybrid` (14.88s vs 7.68s)

## Quality (higher is better, except abstained)

| pipeline | hit@5 | recall@5 | mrr | ndcg@5 | doc_hit@5 | faithfulness | answer_relevance | correctness | abstained |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.917 | 0.917 | 0.854 | 0.869 | **1.000** | **0.963** | 0.917 | 0.854 | 0.083 |
| vector | 0.833 | 0.833 | 0.750 | 0.772 | **1.000** | 0.955 | **0.958** | **0.875** | 0.000 |
| hybrid | **1.000** | **1.000** | **0.892** | **0.918** | **1.000** | 0.948 | 0.917 | 0.854 | 0.083 |
| agentic | 0.833 | 0.833 | 0.736 | 0.761 | 0.917 | 0.932 | 0.875 | 0.812 | 0.167 |

## Latency and cost (lower is better)

| pipeline | retrieval_s_p50 | retrieval_s_p95 | total_s_p50 | total_s_p95 | llm_calls_mean | prompt_mean | completion_mean |
|---|---|---|---|---|---|---|---|
| bm25 | **0.00** | **0.00** | 9.69 | 12.04 | **1.00** | 1227.42 | 106.83 |
| vector | 0.02 | 0.04 | 8.57 | 12.04 | **1.00** | 1228.50 | **101.08** |
| hybrid | 0.02 | 0.02 | **7.68** | **10.34** | **1.00** | **1216.83** | 104.08 |
| agentic | 3.14 | 17.77 | 14.88 | 25.24 | 2.67 | 1813.42 | 153.08 |

## hit@5 by question type

| question type | bm25 | vector | hybrid | agentic |
|---|---|---|---|---|
| conceptual | 1.000 | 0.833 | 1.000 | 0.833 |
| keyword | 0.833 | 0.833 | 1.000 | 0.833 |

## Agent behaviour

- First strategy chosen: `semantic` 10, `keyword` 2
- Average retrieve→grade attempts: 1.17
- Context accepted by grader: 11/12 (92%); the rest fell back to fusing all attempts

Routing by question type (first strategy chosen):

| type | keyword | semantic |
|---|---|---|
| conceptual | 0 | 6 |
| keyword | 2 | 4 |

- Grading decisions: consensus 8, llm 6

Questions where the agent missed the gold chunk (up to 5):

- **q002** (conceptual): What role does a TLS Termination Proxy play in a containerized environment? — strategies semantic → balanced
- **q009** (keyword): How do I start my FastAPI app for development? — strategies semantic

## Metric definitions

- **Hit@5 / Recall@5 / MRR / nDCG@5**: was the gold chunk retrieved, and how high.
- **Doc-Hit@5**: looser; any chunk from the gold document counts.
- **Faithfulness**: share of answer claims supported by the retrieved context (LLM judge).
- **Answer relevance / correctness**: LLM judge 1–5 scores rescaled to 0–1.
- **Abstained**: share of answers that said "I don't know".
