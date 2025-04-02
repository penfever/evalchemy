MODEL_NAME="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
#TASKS="MTBench,MTBench,MATH500,GPQADiamond,AIME24,AIME25,alpaca_eval,leaderboard,LiveCodeBench"
TASKS="MTBench"
ANNOTATOR="gpt-4o-mini-2024-07-18"

MY_COMMAND="python -m eval.eval \
            --model vllm \
            --tasks $TASKS \
            --model_args pretrained=$MODEL_NAME,dtype=bfloat16,gpu_memory_utilization=0.8,max_model_len=16384 \
            --reasoning-postproc \
            --output_path logs \
            --annotator_model $ANNOTATOR"


eval "$MY_COMMAND"