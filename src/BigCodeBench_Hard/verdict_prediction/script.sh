cd src/BigCodeBench_Hard/verdict_prediction

GENERATED_CODE=../actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_code_generate.json
MODEL=qwen3-coder-30B-A3B-instruct

python run_verdict_prediction.py --generated-code $GENERATED_CODE --method direct_verdict   --model $MODEL
