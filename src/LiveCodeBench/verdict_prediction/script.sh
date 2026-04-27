cd src/LiveCodeBench/verdict_prediction

python run_verdict_prediction.py \
  --method direct_verdict \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --model-name qwen3-coder-30B-A3B-instruct \
  --base-url http://129.254.222.36:8008/v1 \
  --temperature 0.8

python run_verdict_prediction.py \
  --method reasoned_verdict \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --model-name qwen3-coder-30B-A3B-instruct \
  --base-url http://129.254.222.36:8008/v1 \
  --temperature 0.8

python run_verdict_prediction.py \
  --method failure_analysis \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --model-name qwen3-coder-30B-A3B-instruct \
  --base-url http://129.254.222.36:8008/v1 \
  --temperature 0.8
