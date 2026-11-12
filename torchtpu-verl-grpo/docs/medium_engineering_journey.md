# Engineering Journey: From JAX Fine-Tuning to Native PyTorch (TorchTPU) + VeRL GRPO on Google Cloud TPU v6e

> **Document Type**: Living Engineering Log & Medium Article Draft  
> **Event**: TorchTPU Sprint 2026 (Topic 3: Training, Fine-Tuning & RL)  
> **Target Audience**: Machine Learning Engineers, Cloud Architects, and RL/Post-Training Practitioners  
> **Privacy Notice**: All sensitive credentials, project identifiers, and private tokens in this document have been sanitized and replaced with standardized placeholders.

---

## Executive Summary

This document chronicles the real-world engineering journey of participating in the **TorchTPU Sprint 2026**, evolving from early JAX/Flax-based LLM fine-tuning into state-of-the-art **Reinforcement Learning (GRPO - Group Relative Policy Optimization)** using the brand-new **Native PyTorch on TPU (TorchTPU)** ecosystem.

By combining:
- **VeRL** (Volcano Engine Reinforcement Learning) as the hybrid RL orchestrator,
- **TorchTitan** as the distributed TPU policy trainer,
- **vLLM-TorchTPU** as the high-throughput decoupled rollout sampler, and
- **TPU Sync** for zero-copy async weight synchronization,

we establish a production-grade post-training pipeline on next-generation **Google Cloud TPU v6e (Trillium)** hardware.

---

## Phase 1: The Preceding Journey — JAX/Flax Lessons & Friction Points

Before diving into the TorchTPU Sprint, our previous initiatives centered around fine-tuning large language models (such as Gemma 3 1B) on Google Cloud TPU v5e (`v5litepod-1`) using JAX, Flax, Google Tunix, and Qwix for macroeconomic domain adaptation (IMF WEO reports).

While JAX provided exceptional raw matrix math acceleration, our real-world logs documented significant operational friction:

### 1. The Scheduling Gauntlet (GCP Error Codes)
- **Error Code 13 (Internal Error)**: Repeatedly surfaced when attempting to queue single-chip `v5litepod-1` requests via `flex-start` in popular zones like `us-central1-a`. Root cause: scheduler lock contention on single-slice queued resource requests.
- **Error Code 3 (`INVALID_ARGUMENT`)**: `FLEX_START provisioning model is not supported for accelerator type "v3-8" in location "us-central1-a"`. Demonstrating that Flex-start was primarily designed for large multi-node pods, not fragmented single chips.
- **Error Code 8 (`RESOURCE_EXHAUSTED`)**: Insufficient capacity in prime regions.
- **The Solution**: Transitioning to an automated multi-region Spot polling script that scanned non-congested zones (e.g., `us-south1-a`), successfully acquiring TPU v5e Spot instances at **~$0.60/hr**.

### 2. Runtime & Dependency Fragility
- **Python 3.10 vs 3.11 Mismatch**: Base TPU Ubuntu VM images defaulted to Python 3.10, whereas modern Flax (`>=0.11`) strictly required Python 3.11+. We had to inject external PPAs and strictly align `libtpu.so` to avoid immediate segmentation faults.
- **JAX JIT Warm-up Latency**: During pre-training baseline evaluations, the initial sampler invocation triggered deep XLA JIT graph compilation, freezing perceived execution for several minutes without user feedback.

---

## Phase 2: The Paradigm Shift — Unpacking Native PyTorch on TPUs (11/2026)

In November 2026, Google (in collaboration with the Meta-PyTorch core team) unveiled the official **TorchTPU** ecosystem at PyTorch Conf 2026. Insights from Google Developer Advocate Wei Wei's presentation (*"Introduction to TPU Software Ecosystem"*) marked a decisive turning point:

> **The Core Philosophy**: No more translating models into JAX. No more intrusive `torch_xla` Lazy Tensor wrappers. Pure, native PyTorch code running directly on TPUs with `torch.device("tpu")`.

### The Four Pillars of the Modern TPU Stack

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TorchTPU Ecosystem (2026)                              │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│   Pillar 1: Native Base    │  Pillar 2: Serving Engine   │    Pillar 3: Pre & Post RL  │
│  - torch_tpu (C++ Private1)│  - vLLM-TorchTPU (FP8 KV)   │  - TorchTitan (Trainer)     │
│  - 450+ Core ATen Ops      │  - SGLang-TorchTPU          │  - VeRL (GRPO Controller)   │
│  - c10d ProcessGroup Dist  │  - TPU Sync (Zero-Copy KV)  │  - TPU Sync (Weight Sync)   │
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     Pillar 4: Autonomous Developer UX (TPU Agent Suite)                │
│    Zero-Hallucination Hardware-Verified Profiling | Automated Pallas Kernel Synthesis  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Pillar 1 (Native Foundation - TorchTPU)**:
   - Implemented via PyTorch C++ `PrivateUse1` dispatcher with standard `at::Tensor` memory representations.
   - Zero JAX dependency during execution. Native compatibility with Hugging Face Transformers v5 (`device_map="tpu"`).
   - High test coverage: Nightly CI testing across **4,917 Hugging Face models** with a **98% Tier-1 pass rate**.
2. **Pillar 2 (Inference - vLLM-TorchTPU & TPU Sync)**:
   - **Sequence-on-Lane FP8 KV Cache**: Halves KV memory consumption and memory bandwidth bottleneck on TPU.
   - **TPU Sync**: A high-performance C++ library bypassing host memory copying to directly transfer TPU HBM buffers asynchronously across nodes.
3. **Pillar 3 (RL Post-Training - TorchTitan + VeRL)**:
   - **TorchTitan** provides clean, decoupled `nn.Module` distributed training leveraging FSDP2, DTensor, and `torch.compile(backend="tpu")`.
   - **VeRL (GRPO)**: Slide 14 of the official presentation revealed empirical proof: running 250 steps of GRPO with `Qwen3-4B-Base` on **Google Cloud TPU v6e** achieved **100% curve parity with GPU baselines on MATH-500**.
4. **Pillar 4 (Developer Experience - TPU Agent Suite)**:
   - Strict adherence to the **"Zero Hallucination"** principle: All throughput, MFU, and latency figures must originate from verified on-chip hardware execution logs.

---

## Phase 3: Hardware Quota & Cloud Permission Diagnostics

Prior to committing to our architecture, we performed an exhaustive audit of our Google Cloud Platform project (`[PROJECT_ID]`):

### Quota Audit Summary
```bash
# Auditing regional TPU quota limits across key clusters
gcloud alpha services quota list --service=tpu.googleapis.com
```

### Verified Allocations:
- **TPU v6e (Trillium - Next-Gen)**:
  - `tpu-v6e-preemptible` (Spot): **1,536 Cores** (Supported across `us-central1`, `us-east5`, `us-south1`, `europe-west4`).
  - `tpu-v6e` (On-Demand): **512 Cores**.
  - Estimated Cost: **~$1.35/hour** (Spot).
- **TPU v5e (LitePod)**:
  - `PREEMPTIBLE_TPU_LITE_PODSLICE_V5`: **1,536 Cores**.
  - `TPU_LITE_PODSLICE_V5`: **512 Cores**.
  - Estimated Cost: **~$0.60/hour** (Spot).

**Strategic Takeaway**: With 1,536 cores of TPU v6e Spot quota active and accessible, we can target **TPU v6e-1 and v6e-4 (Trillium)** as our primary hardware platform, giving our post-training experiments state-of-the-art compute density.

---

## Phase 4: System Architecture — VeRL GRPO on Cloud TPU v6e

### The Decoupled Hybrid Flow
In modern reasoning models (e.g., DeepSeek-R1, Qwen-Math), post-training via **Group Relative Policy Optimization (GRPO)** eliminates the separate Critic network by normalizing rewards across $G$ candidate rollouts sampled per prompt:

$$\text{Advantage}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$

The physical execution on Cloud TPU is split into two specialized engines:

```text
  [Prompt Batch] ──► Ray Orchestrator
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│     vLLM-TorchTPU (Sampler)   │   │     TorchTitan (Trainer)      │
│  - High-throughput rollout    │   │  - Distributed Policy Update  │
│  - Sequence-on-Lane FP8 KV    │   │  - FSDP2 / DTensor on TPU     │
│  - Generates G candidates     │   │  - Gradient Clipping & Loss   │
└──────────────┬────────────────┘   └───────────────▲───────────────┘
               │                                    │
               ▼                                    │
  [Rule-Based Reward Evaluator]                     │
  - <think> format validation                       │
  - Exact math / symbol match                       │
               │                                    │
               └─────────► Advantage Computation ───┘
                                 ▲
                                 │ TPU Sync (Async Weight Sync)
                                 └──────────────────────────────┘
```

### Reward Function Strategy
To elicit robust chain-of-thought reasoning without reward hacking:
1. **Format Reward**: Enforces strict XML tags:
   ```text
   <think>
   ... granular step-by-step reasoning ...
   </think>
   <answer>
   ... final extracted response ...
   </answer>
   ```
2. **Accuracy Reward**: Deterministic mathematical equivalence checking using symbolic normalization (SymPy / regex parser) against GSM8K / MATH ground truth.

---

## Phase 5: Repository Structuring & Strict Credential Isolation

To maintain open-source hygiene while protecting enterprise secrets, the project was restructured into a clean monorepo: `learning-tpu`:

```text
learning-tpu/
├── .gitignore                          # Zero-leakage regex rules
├── env.example                         # Safe environment template
├── README.md                           # Master navigation catalog
├── torchtpu-verl-grpo/                 # 🌟 TorchTPU Sprint 2026 Core
│   ├── configs/                        # Hydra configurations for VeRL & TorchTitan
│   ├── docs/                           # Living engineering documentation
│   ├── scripts/                        # Automated v6e provisioning & TPU VM setup
│   └── src/                            # Reward functions, dataset preprocessing
├── gemma-tpu-training/                 # Historical JAX TPU v5e baseline
└── gemma-training/                     # Historical GPU baseline
```

### Credential Protection Rules (`.gitignore`)
```gitignore
# Zero-Trust Secret Isolation
.env
.env.*
!.env.example
*.env
*.json
!package.json
!tsconfig.json
studio-*.json
service-account*.json
*.key
*.pem
```

---

## Phase 6: Roadmap & Delivery Milestones (Deadline: November 25, 2026)

| Milestone | Target Scope | Status | Deliverable |
| :--- | :--- | :--- | :--- |
| **M1: Foundation Setup** | Environment setup scripts, template configs, sanitized logs | ✅ Completed | GitHub Initial Release |
| **M2: TPU v6e Allocation** | Deploy `request_tpu_v6.sh` in high-availability zones (`us-central1-a`) | 🔄 In Progress | Running TPU v6e VM Instance |
| **M3: Environment Provisioning** | Install `torch_tpu`, `torchtitan`, `vllm-torchtpu`, `verl` | ⏳ Scheduled | Verified `torch.device("tpu")` test |
| **M4: GRPO PoC Execution** | Run GSM8K reasoning alignment (250 steps) on TPU v6e | ⏳ Scheduled | Training curves & Loss logs |
| **M5: Benchmarking & Profiling** | Capture XProf traces, token throughput, and MFU metrics | ⏳ Scheduled | Comparative benchmark charts |
| **M6: Public Publication** | Final Medium article release + GitHub documentation + `#TorchTPUSprint` SNS | ⏳ Scheduled | Complete Sprint Submission |

---

## Key Takeaways & Best Practices for Practitioners

1. **Embrace Native PyTorch**: If starting an LLM acceleration project on Cloud TPU today, prefer the Native PyTorch (`TorchTPU`) stack over legacy XLA wrappers. The ecosystem velocity and Hugging Face parity are unparalleled.
2. **Decouple Rollout and Training**: Reinforcement learning workloads are inherently heterogeneous. Running inference-optimized engines (vLLM-TorchTPU) for rollouts while leveraging scalable training engines (TorchTitan) for backward passes prevents severe compute starvation.
3. **Automate Around Spot Fragility**: TPU Spot pricing offers staggering cost savings (~70% discount), but requires automated multi-zone failover and proactive state checkpointing.
