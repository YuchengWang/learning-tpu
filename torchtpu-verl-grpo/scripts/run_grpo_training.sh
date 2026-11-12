#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Activate Environment
ENV_DIR="$HOME/torchtpu_env"
if [ -d "$ENV_DIR" ]; then
    source "$ENV_DIR/bin/activate"
fi

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "=========================================================="
echo "🔥 Starting VeRL GRPO Training on Cloud TPU"
echo "Stack: VeRL + TorchTitan (Trainer) + vLLM-TorchTPU (Sampler)"
echo "Config: configs/grpo_torchtitan_vllm_tpu.yaml"
echo "=========================================================="

# 2. Check/Initialize Ray Cluster
echo "⚡ Step 1: Checking Ray Cluster status..."
if ! ray status &> /dev/null; then
    echo "Starting local Ray head node..."
    ray start --head --disable-usage-stats
fi

# 3. Ensure Dataset Prepared
if [ ! -f "data/mbpp/train.parquet" ]; then
    echo "📊 Step 2: Preparing Google Research MBPP Parquet dataset..."
    python3 src/data_preprocess.py --output_dir data/mbpp
fi

# 4. Launch VeRL GRPO Training
echo "🚀 Step 3: Launching VeRL GRPO Main Pipeline..."
mkdir -p outputs/logs

python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs" \
    --config-name='grpo_torchtitan_vllm_tpu' \
    2>&1 | tee outputs/logs/grpo_training_$(date +%Y%m%d_%H%M%S).log

echo "=========================================================="
echo "✅ Training Completed!"
echo "Checkpoints saved in outputs/grpo_checkpoints"
echo "=========================================================="
