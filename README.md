# Learning TPU & Accelerated Computing

A comprehensive collection of practical engineering implementations, deployment pipelines, benchmarks, and tutorials for fine-tuning and post-training large language models on **Google Cloud TPUs** and high-performance accelerators.

---

## 🗂️ Project Catalog

| Subproject | Hardware | Framework / Stack | Description |
| :--- | :--- | :--- | :--- |
| **[`torchtpu-verl-grpo`](./torchtpu-verl-grpo)** | **TPU v6e / v5e** | **Native TorchTPU + VeRL + TorchTitan + vLLM-TorchTPU** | **TorchTPU Sprint 2026**: Post-training & Reinforcement Learning with GRPO, Google Gemma 4, TorchTitan distributed trainer, and vLLM-TorchTPU inference engine. |
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

## 🔐 Environment Configuration & Cloud Credentials

This repository interacts directly with Google Cloud Platform (Cloud TPU, Vertex AI, and Compute Engine).

### 1. Configuration Setup
A template configuration file [`env.example`](./env.example) is provided. To set up your local environment:
```bash
cp env.example .env
```
Edit `.env` with your active project details:
- `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (e.g. `studio-9772401964-cc157`).
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your GCP Service Account JSON key (e.g. `./studio-9772401964-cc157-f61ab654b16f.json`).
- `HF_TOKEN`: Your Hugging Face user access token (for gated models like Gemma).
- `ZONE` and `ACCELERATOR_TYPE`: Cloud TPU cluster settings (e.g., `us-central1-a`, `v6e-1` or `v5litepod-1`).

### 2. ⚠️ Security Notice (Strictly Enforced)
> [!IMPORTANT]
> **Credential Privacy & Git Protection**:
> - Real environment files (`.env`, `*.env`) and GCP Service Account private keys (`studio-*.json`, `*.json`, `*.key`, `*.pem`) contain sensitive cloud credentials and are **strictly ignored** via [`.gitignore`](./.gitignore).
> - **Never commit or push `.env` or service account JSON key files to any remote Git repository.**
> - Use [`env.example`](./env.example) as the sole version-controlled template for environment variable schemas.

---

## 🚀 Quick Navigation

- [TorchTPU Sprint 2026: VeRL + TorchTitan + vLLM Guide](./torchtpu-verl-grpo/README.md)
- [Gemma 3 TPU Deployment & Reproduction Guide](./gemma-tpu-training/README.md)
- [Gemma 3 TPU Engineering Blog (CN)](./gemma-tpu-training/Medium_TPU_Gemma3_Blog_Standalone_CN.md)
- [Gemma 3 TPU Engineering Blog (EN)](./gemma-tpu-training/Medium_TPU_Gemma3_Blog_Standalone_EN.md)

---

## 📄 License
This repository is licensed under the [Apache-2.0 License](./LICENSE).
