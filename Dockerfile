FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY LiveCodeBench/ /workspace/LiveCodeBench/
COPY BigCodeBench_Hard/ /workspace/BigCodeBench_Hard/
