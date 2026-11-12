# Post-training & Reinforcement Learning with VeRL on Google Cloud TPUs

> **TorchTPU Sprint 2026 Project**  
> **Author**: Yucheng Wang (AI GDE)  
> **Track**: Topic 3 — Training, Fine-tuning & RL (VeRL + TorchTitan + vLLM-TorchTPU)

---

## 📌 Project Overview

This project implements and optimizes post-training and reinforcement learning workflows on next-generation **Google Cloud TPUs (v6e Trillium / v5e)**, specifically focusing on:

- **Base Model**: **Google Gemma 4** (Lightweight `google/gemma-4-E2B-it`, Google's 2026 flagship open model family).
- **Dataset**: **Google Research MBPP** (Mostly Basic Python Problems, with unit-test assertion verification).
- **Algorithm**: **GRPO** (Group Relative Policy Optimization) chain-of-thought code reasoning alignment via **VeRL (Hybrid Engine)**.
- **Distributed Trainer**: **TorchTitan** (leveraging native PyTorch `torch.device("tpu")`, SimpleFSDP, DTensor, SplashAttention, and `torch.compile` for distributed policy updates).
- **Inference & Rollout Engine**: **vLLM-TorchTPU** (delivering ultra-high-throughput distributed sampling with Sequence-on-Lane FP8 KV caching).
- **Weight Synchronization**: **TPU Sync** (C++ high-performance library for zero-copy async weight synchronization between rollout and training engines).
- **Dependency Hygiene**: 100% pinned package versions and Git Release tags (`torch_tpu==0.2.0`, `torchtitan==0.3.0`, `verl@v0.9.1`, `verl-hardware-plugin@v0.1.0`).

---

## 🚀 System Architecture

```text
                               ┌────────────────────────────────────────────────────────┐
                               │       VeRL Hybrid Controller & Ray Orchestrator        │
                               │           (GRPO: Group Relative Policy Optimization)   │
                               └──────────────┬───────────────────────────┬─────────────┘
                                              │                           │
                                (1) Prompt Rollout & Sampling  (2) Policy Updates & Loss Backprop
                                              ▼                           ▼
                      ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
                      │          vLLM-TorchTPU          │   │           TorchTitan            │
                      │    Decoupled Rollout Engine     │   │     Distributed Policy Trainer  │
                      │     - PagedAttention on TPU     │   │     - Native torch.device("tpu")│
                      │     - Sequence-on-Lane FP8 KV   │   │     - FSDP2 / DTensor           │
                      │     - Generates G candidates    │   │     - SPMD Collective Comm      │
                      └────────────────┬────────────────┘   └─────────────▲───────────────────┘
                                       │                                  │
                                       ▼                                  │
                          [Rule-Based Reward Evaluator]                   │
                          - <think> format validation                     │
                          - Numerical & symbolic math check               │
                                       │                                  │
                                       └──────────► Advantage ────────────┘
                                                         ▲
                                                         │ TPU Sync (Async Weight Sync)
                                                         └────────────────┘
```

---

## 📁 Repository Structure

```text
torchtpu-verl-grpo/
├── configs/
│   └── grpo_torchtitan_vllm_tpu.yaml     # VeRL GRPO configuration (TorchTitan + vLLM)
├── docs/
│   └── medium_engineering_journey.md     # In-depth technical log & sanitized Medium draft
├── scripts/
│   ├── request_tpu_v6.sh                 # Automated TPU v6e Spot provisioning with multi-zone fallback
│   ├── monitor_tpu.sh                    # TPU VM & Queued Resource status monitor
│   ├── sync_to_tpu.sh                    # One-click code and configuration push to Cloud TPU VM
│   ├── setup_tpu_env.sh                  # Remote TPU VM bootstrap (Python 3.11, TorchTPU, Ray, VeRL)
│   └── run_grpo_training.sh              # Distributed GRPO training launcher
├── src/
│   ├── check_tpu.py                      # Hardware & TorchTPU environment health check
│   ├── data_preprocess.py                # Google Research MBPP dataset converter to VeRL Parquet
│   └── reward_functions.py               # R1-style format & mathematical accuracy reward functions
├── env.example                           # Sanitized configuration template
├── requirements.txt                      # Project dependencies
└── README.md
```

---

## 🛠️ Step-by-Step Execution Guide

### Phase 1: Local Preparation & TPU Provisioning
1. **Configure Environment**:
   ```bash
   cp env.example .env
   # Edit .env with your PROJECT_ID and settings
   ```

2. **Provision TPU v6e Spot Instance**:
   ```bash
   bash scripts/request_tpu_v6.sh
   ```
   *Note: The script attempts direct Spot creation across verified high-availability zones (`us-central1-a`, `us-east5-a`, `us-south1-a`).*

---

### Phase 2: Project Synchronization
Push your local code, configurations, and scripts to the active TPU VM:
```bash
bash scripts/sync_to_tpu.sh <NODE_ID> <ZONE>
```

---

### Phase 3: Remote Environment Setup
SSH into the Cloud TPU VM and execute the automated environment bootstrap:
```bash
gcloud compute tpus tpu-vm ssh <NODE_ID> --zone <ZONE>

# On the TPU VM:
cd ~/torchtpu-verl-grpo
bash scripts/setup_tpu_env.sh
```
*This installs Python 3.11, TorchTPU, TorchTitan, vLLM-TorchTPU, and VeRL, followed by a hardware verification check.*

---

### Phase 4: Launching GRPO Post-Training
Run the end-to-end training pipeline:
```bash
# On the TPU VM:
source ~/torchtpu_env/bin/activate
cd ~/torchtpu-verl-grpo

bash scripts/run_grpo_training.sh
```

---

## ⚠️ Important: Resource Teardown
To avoid unwanted charges, always delete the TPU instance once your experiment finishes:
```bash
gcloud compute tpus tpu-vm delete <NODE_ID> --zone <ZONE> --quiet
```

---

## 📖 In-Depth Technical Log
For a comprehensive breakdown of the engineering post-mortem, GCP error codes, architecture decisions, and PyTorch Conf 2026 insights, read:
- **[Living Engineering Journey & Medium Draft](./docs/medium_engineering_journey.md)**

---

## 📄 License & Attribution
Licensed under Apache 2.0. Developed by Yucheng Wang for the **TorchTPU Sprint 2026**.
