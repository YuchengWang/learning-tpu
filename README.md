# Learning TPU & Accelerated Computing

A comprehensive collection of practical engineering implementations, deployment pipelines, benchmarks, and tutorials for fine-tuning and post-training large language models on **Google Cloud TPUs** and high-performance accelerators.

---

## 🗂️ Project Catalog

| Subproject | Hardware | Framework / Stack | Description |
| :--- | :--- | :--- | :--- |
| **[`torchtpu-verl-grpo`](./torchtpu-verl-grpo)** | **TPU v6e / v5e** | **PyTorch/XLA + VeRL + TorchTitan + vLLM-TorchTPU** | **TorchTPU Sprint 2026**: Post-training & Reinforcement Learning with GRPO, TorchTitan distributed trainer, and vLLM-TorchTPU inference engine. |
| **[`gemma-tpu-training`](./gemma-tpu-training)** | **TPU v5e** (`v5litepod-1`) | **JAX / Flax + Google Tunix + Qwix** | QLoRA fine-tuning of Gemma 3 1B on Cloud TPU v5e with IMF macroeconomic data and automated spot instance provisioning. |
| **[`gemma4-tpu-training`](./gemma4-tpu-training)** | **TPU v5e** | **JAX / Flax + Google Tunix + Qwix** | Experimental QLoRA fine-tuning of Gemma 4 e2b on Cloud TPU. |
| **[`gemma-training`](./gemma-training)** | **GPU** (NVIDIA CUDA) | **PyTorch / Hugging Face PEFT** | GPU-based baseline QLoRA fine-tuning workflow for Gemma models. |

---

## 🌟 Highlights

1. **Production-Ready Cloud TPU Automation**:
   - Zero-downtime Spot TPU scanning and automated capacity acquisition.
   - Remote environment setup scripts handling `libtpu`, Python 3.11 PPA, and XLA runtime flags.
2. **Modern Reinforcement Learning on TPUs**:
   - State-of-the-art GRPO (Group Relative Policy Optimization) leveraging VeRL hybrid engine.
   - Decoupled high-throughput rollout sampling with vLLM-TorchTPU and distributed policy updates with TorchTitan.
3. **In-depth Engineering Post-Mortems**:
   - Step-by-step guides for troubleshooting TPU scheduling error codes, memory allocation tricks (`XLA_PYTHON_CLIENT_MEM_FRACTION`), and JAX JIT compilation caching.

---

## 🚀 Quick Navigation

- [TorchTPU Sprint 2026: VeRL + TorchTitan + vLLM Guide](./torchtpu-verl-grpo/README.md)
- [Gemma 3 TPU Deployment & Reproduction Guide](./gemma-tpu-training/README.md)
- [Gemma 3 TPU Engineering Blog (CN)](./gemma-tpu-training/Medium_TPU_Gemma3_Blog_Standalone_CN.md)
- [Gemma 3 TPU Engineering Blog (EN)](./gemma-tpu-training/Medium_TPU_Gemma3_Blog_Standalone_EN.md)

---

## 📄 License
This repository is licensed under the [Apache-2.0 License](./LICENSE).
