# Hugging Face Space (Docker SDK) for the Agentic RAG Benchmarking System.
#
# On Hugging Face the Space repo only contains this Dockerfile + README.md, so the code is
# cloned from GitHub (REPO/REF). For a local test build, run from the project root:
#   docker build -f deploy/huggingface/Dockerfile -t agentic-rag-space .
#   docker run --rm -p 7860:7860 agentic-rag-space
# and the local source tree is used instead of cloning.
#
# Layer order = least to most frequently changing (Ollama, model, torch, then code), so a code
# change only re-runs the last few steps.
FROM python:3.11-slim

ARG REPO=https://github.com/neelkapadi/Agentic-RAG-Benchmarking-System.git
ARG REF=main
# Small model: free Spaces have 2 vCPUs and no GPU.
ARG MODEL=qwen2.5:1.5b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/ollama-models \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    DEMO_MODE=1 \
    RAG_LLM_MODEL=$MODEL \
    RAG_JUDGE_MODEL=$MODEL \
    RAG_AGENT_MAX_ATTEMPTS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face runs containers as uid 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Bake the model into the image (needs a temporary Ollama server during the build).
RUN ollama serve > /tmp/ollama-build.log 2>&1 & \
    for i in $(seq 1 60); do curl -sf http://localhost:11434/api/tags > /dev/null && break; sleep 1; done \
    && ollama pull "$MODEL"

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

# Local build context contains the source tree; on Hugging Face it only has Dockerfile + README.
COPY --chown=user . /tmp/context
RUN if [ -f /tmp/context/src/pipeline.py ]; then cp -r /tmp/context/. . ; \
    else git clone --depth 1 --branch "$REF" "$REPO" . ; fi \
    && rm -rf /tmp/context \
    && pip install --user -r requirements.txt

# Docs + chunks + vector index are built into the image so startup is fast.
RUN python -m scripts.download_docs && python -m scripts.build_index

EXPOSE 7860
CMD ["bash", "deploy/huggingface/start.sh"]
