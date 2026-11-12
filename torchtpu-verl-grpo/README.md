# Post-training & Reinforcement Learning with VeRL on Google Cloud TPUs

> **TorchTPU Sprint 2026 Project**  
> **Author**: Yucheng Wang (AI GDE)  
> **Track**: Training, Fine-tuning & RL (VeRL + TorchTitan + vLLM-TorchTPU)

---

## 📌 Project Overview

This project implements and optimizes post-training and reinforcement learning workflows on Google Cloud TPUs (v5e / v6e Trillium), specifically focusing on:

- **Algorithm**: **GRPO** (Group Relative Policy Optimization) / RLHF via **VeRL (Hybrid Engine)**.
- **Distributed Trainer**: **TorchTitan** (leveraging PyTorch/XLA, FSDP, DTensor, and `torch.compile` for distributed TPU training).
- **Inference & Rollout Engine**: **vLLM-TorchTPU** (delivering high-throughput distributed sampling during RL rollouts).

---

## 🚀 Key Milestones & Architecture

```text
       ┌─────────────────────────────────────────────────────────────┐
       │                VeRL Orchestrator (Ray / Core)               │
       └──────────────┬───────────────────────────────┬──────────────┘
                      │                               │
         (1) Rollout / Sampling          (2) Policy & Critic Update
                      ▼                               ▼
       ┌─────────────────────────────┐ ┌─────────────────────────────┐
       │       vLLM-TorchTPU         │ │         TorchTitan          │
       │   High-Throughput Sampling  │ │  Distributed TPU Trainer    │
       │  (Cloud TPU v5e/v6e Pod)    │ │   (FSDP / DTensor / XLA)    │
       └─────────────────────────────┘ └─────────────────────────────┘
```

---

## 📁 Directory Layout

- `configs/`: Hyperparameters and cluster configurations for VeRL, TorchTitan, and vLLM.
- `scripts/`: Shell scripts for TPU provisioning, environment setup, and distributed launch.
- `src/`: Core entry points, reward functions, and dataset loading pipelines.
- `docs/`: Technical reports, benchmarks, and sprint tutorials.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Google Cloud SDK (`gcloud`) authenticated with TPU quota.
- Cloud TPU v5e or v6e instance.
- Python 3.11+ environment with PyTorch/XLA and Ray.

### 2. Environment Setup
```bash
bash scripts/setup_env.sh
```

### 3. Launching Training
```bash
bash scripts/run_grpo_training.sh
```

---

## 📄 License & Attribution
Apache 2.0. Developed as part of TorchTPU Sprint 2026.
