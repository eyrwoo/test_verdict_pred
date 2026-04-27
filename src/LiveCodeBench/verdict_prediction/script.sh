cd src/LiveCodeBench/verdict_prediction

python run_verdict_prediction.py \
  --method direct_verdict \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --model-name qwen3-coder-30B-A3B-instruct \
  --base-url http://localhost:8008/v1 \
  --temperature 0
