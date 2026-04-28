FROM python:3.10-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/workspace/src:/workspace/src/LiveCodeBench/actual_exec/LiveCodeBench:/workspace/src/BigCodeBench_Hard/actual_exec/bigcodebench

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/

RUN cp /workspace/src/BigCodeBench_Hard/actual_exec/bcb_overrides/eval/__init__.py \
       /workspace/src/BigCodeBench_Hard/actual_exec/bigcodebench/bigcodebench/eval/__init__.py \
    && cp /workspace/src/BigCodeBench_Hard/actual_exec/bcb_overrides/evaluate.py \
       /workspace/src/BigCodeBench_Hard/actual_exec/bigcodebench/bigcodebench/evaluate.py

CMD ["bash"]
