FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

RUN hf download intfloat/multilingual-e5-small \
    --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
    --include "*.json" "model.safetensors" "sentencepiece.bpe.model" \
    --local-dir /opt/models/multilingual-e5-small

RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 util-linux \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

COPY app.py config.py documents.py ingest.py rag.py ./
USER 10001
