#!/usr/bin/env bash
# Container entrypoint: start Ollama, warm up the model, then serve the dashboard on 7860.
set -euo pipefail

ollama serve > /tmp/ollama.log 2>&1 &

for i in $(seq 1 60); do
  curl -sf http://localhost:11434/api/tags > /dev/null && break
  sleep 1
done

# Load the model into memory so the first visitor does not wait for it.
curl -sf http://localhost:11434/api/generate \
  -d "{\"model\": \"${RAG_LLM_MODEL}\", \"prompt\": \"hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
  > /dev/null || echo "warm-up failed; continuing"

exec streamlit run dashboard/streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.maxUploadSize 10 \
  --client.toolbarMode viewer
