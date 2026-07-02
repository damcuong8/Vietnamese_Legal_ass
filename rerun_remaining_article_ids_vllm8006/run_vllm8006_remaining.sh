#!/usr/bin/env bash
set -euo pipefail

cd /home/uet/cuongdam/Legal_assistant

IDS="$(paste -sd, rerun_remaining_article_ids_vllm8006/ids.txt)"

env   CUDA_VISIBLE_DEVICES=0   LLM_BASE_URL=http://127.0.0.1:8006/v1   LLM_MODEL=qwen3.5-9b   LLM_API_KEY=EMPTY   LLM_MAX_TOKENS=65536   CHECKPOINTER_BACKEND=none   COMPRESS_LLM_ENABLE_THINKING=false   EMBEDDING_MAX_CONCURRENT=1   RERANKER_MAX_CONCURRENT=1   RERANKER_BATCH_SIZE=32   python test_a.py     --llm-backend default     --ids "$IDS"     --workers 12     --graph-mode stateless     --out-dir /home/uet/cuongdam/Legal_assistant/rerun_remaining_articles_vllm8006     --results-name results.json     --no-zip
