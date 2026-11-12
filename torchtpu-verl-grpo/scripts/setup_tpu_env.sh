#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Bootstrapping TorchTPU Ecosystem on Cloud TPU VM"
echo "=========================================================="

# 1. Update OS and Install Python 3.11 if needed
echo "📦 Step 1: Updating system packages and installing Python 3.11..."
sudo apt-get update -y
sudo apt-get install -y software-properties-common curl git build-essential

if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils
fi

# 2. Setup Dedicated Virtual Environment
ENV_DIR="$HOME/torchtpu_env"
if [ ! -d "$ENV_DIR" ]; then
    echo "🐍 Step 2: Creating virtual environment at $ENV_DIR..."
    python3.11 -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# 3. Install TorchTPU Stack (Native PyTorch on TPUs)
echo "🔥 Step 3: Installing Native PyTorch & TorchTPU plugin..."
pip install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Install Project Requirements
echo "📚 Step 4: Installing base dependencies from requirements.txt..."
cd "$(dirname "$0")/.."
pip install -r requirements.txt

# 5. Install Ecosystem Frameworks (Strictly Pinned Versions)
echo "⚡ Step 5: Installing TorchTitan, VeRL and vLLM for TPU (Pinned Releases)..."
VLLM_TARGET_DEVICE=tpu pip install vllm-tpu==0.21.0 || true
pip install torchtitan==0.3.0 spmd_types==0.2.3 torch_checkpointing==0.1.1 tyro==1.0.17 einops==0.8.2 --no-deps
pip install git+https://github.com/verl-project/verl.git@v0.9.1
pip install git+https://github.com/verl-project/verl-hardware-plugin.git@v0.1.0
# Re-align torch_tpu to ensure matching torch and libtpu
pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip uninstall -y torchvision || true

# 6. Run Self-Check
echo "🔍 Step 6: Running TPU Hardware Self-Check..."
python3 src/check_tpu.py

echo "=========================================================="
echo "✅ TorchTPU Environment Setup Completed Successfully!"
echo "   Activate with: source $ENV_DIR/bin/activate"
echo "=========================================================="
