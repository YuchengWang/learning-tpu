# Engineering Journey: From JAX Fine-Tuning to Native PyTorch (TorchTPU) + VeRL GRPO on Google Cloud TPU v6e

*By Yucheng Wang — Google Developer Expert (AI/ML) | TorchTPU Sprint 2026*  
*Open-Source Code & Artifacts: [GitHub Repository: YuchengWang/learning-tpu](https://github.com/YuchengWang/learning-tpu/tree/main/torchtpu-verl-grpo)*

---

![Hero Banner: TorchTPU Sprint 2026](./images/01_hero_banner.png)

If you have ever attempted distributed Reinforcement Learning post-training on Google Cloud TPUs, you are probably intimately familiar with the pain: wrestling with JAX compilation graphs, managing fragile host-device `libtpu` version locks, and watching single-slice Spot instances get silently preempted during an asynchronous rollout step. For years, running PyTorch on TPUs felt like translating idioms across a language barrier using `torch_xla` lazy-tensor wrappers and tracing workarounds.

That changed in November 2026. 

During the **TorchTPU Sprint 2026** (Topic 3: Training, Fine-Tuning & RL), we set out to build and benchmark a production-ready post-training pipeline using the freshly unveiled **Native PyTorch on TPU (TorchTPU)** ecosystem. By pairing **VeRL** (Volcano Engine Reinforcement Learning) as our decoupled orchestrator, **TorchTitan** as our FSDP2 TPU policy trainer, **vLLM-TorchTPU** for high-throughput rollout generation, and **TPU Sync** for asynchronous zero-copy weight synchronization, we established a complete end-to-end Group Relative Policy Optimization (GRPO) workflow on **Google Cloud TPU v6e (Trillium)** hardware.

In this deep dive, we document the architectural decisions, the quota and infrastructure realities of GCP, the underlying C++ dispatcher mechanics of TorchTPU, and the concrete telemetry logs gathered from our test runs.

---

## From JAX to Native PyTorch: The Realities of TPU Workloads

Prior to the TorchTPU Sprint, our distributed LLM fine-tuning pipelines ran on Google Cloud TPU v5e (`v5litepod-1`) chips using JAX, Flax, Google Tunix, and Qwix for domain adaptation on structured economic texts (IMF World Economic Outlook datasets).

While JAX provided blazing-fast dense matrix math, our engineering logs recorded consistent friction in everyday development:

### 1. Scheduler Contention and Error Codes
- **Error Code 13 (`INTERNAL`)**: Surfaced repeatedly when queueing single-slice `v5litepod-1` requests using `flex-start` in congested zones like `us-central1-a`. Under high cluster load, the single-slice queued resource scheduler locked up waiting for fragmented allocations.
- **Error Code 3 (`INVALID_ARGUMENT`)**: `FLEX_START provisioning model is not supported for accelerator type "v3-8" in location "us-central1-a"`. Flex-start allocation was engineered primarily for multi-rack pods, not isolated single-chip nodes.
- **Error Code 8 (`RESOURCE_EXHAUSTED`)**: Common during peak hours across primary US regions.
- **Engineering Fix**: We built an automated multi-region Spot polling script that fell back to uncongested zones (such as `us-south1-a`), reliably acquiring TPU v5e Spot instances at **~$0.60/hr** with automated retry backoffs.

### 2. Runtime and Library Fragility
- **Python Runtime Misalignments**: Standard TPU VM Ubuntu images shipped with Python 3.10, whereas modern Flax releases (`>=0.11`) required Python 3.11+. Manually injecting external PPAs risked breaking system `libtpu.so` drivers, resulting in instant segfaults on device initialization.
- **Compilation Warm-up Delays**: JAX JIT compilation on the initial rollout step would stall execution for several minutes without diagnostic output, making interactive debugging painful.

These operational hurdles highlighted the need for an idiomatic, native PyTorch runtime that behaves like standard PyTorch on modern hardware.

---

## Under the Hood of TorchTPU: C++ Dispatcher and the Four Pillars

At PyTorch Conf in November 2026, Google and the Meta-PyTorch core team introduced the official **TorchTPU** architecture. As detailed in Google Developer Advocate Wei Wei's keynote (*"Introduction to TPU Software Ecosystem"*), this represents an architectural break from legacy execution models.

> **The Design Target**: Eliminate the translation layer. No more converting models to JAX and no more opaque `torch_xla` lazy-tensor graphs. Developers can run pure PyTorch code directly on TPUs using standard device calls: `torch.device("tpu")`.

![The Paradigm Shift: From Legacy JAX/Lazy Tensor to Native TorchTPU (2026)](./images/02_torchtpu_paradigm_shift.png)

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
│    Hardware-Verified Telemetry Profiling | Automated Pallas Kernel Optimization        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**Pillar 1: Native Foundation (TorchTPU)**:
- Integrated directly into PyTorch via the C++ `PrivateUse1` dispatcher key, representing TPU tensors as native `at::Tensor` objects in device HBM.
- Operates without JAX runtime dependencies. Supports standard Hugging Face Transformers v5 workflows (`device_map="tpu"`).
- Verified against **4,917 Hugging Face models** in nightly CI, maintaining a **98% Tier-1 pass rate**.

**Pillar 2: Inference Serving (vLLM-TorchTPU & TPU Sync)**:
- **Sequence-on-Lane FP8 KV Cache**: Employs Pallas TPU kernels to cut KV cache memory footprint by ~48%, unlocking larger rollout batch sizes during generation.
- **TPU Sync**: High-performance C++ transport library bypassing CPU host memory copying to stream TPU HBM buffers directly across workers with sub-15ms transfer latencies.

**Pillar 3: Distributed RL Training (TorchTitan + VeRL GRPO)**:
- **TorchTitan** provides modular distributed training recipes leveraging PyTorch FSDP2, DTensor, and `torch.compile(backend="tpu")`.
- **VeRL Integration**: Enables decoupled Ray workers to coordinate rollout generation and policy optimization. Benchmarks presented on TPU v6e confirmed that 250 steps of GRPO on `Qwen3-4B-Base` matched GPU convergence curves on MATH-500.

**Pillar 4: Developer UX & Tooling (TPU Agent Suite)**:
- Automated hardware telemetry collectors, XProf tracing integration, and automated Pallas TPU kernel synthesis to simplify low-level performance tuning.

---

## GCP Hardware Sizing: Navigating TPU v6e Trillium Quotas and Spot Pricing

Before deploying our RL workloads, we conducted a systematic audit of our Google Cloud Platform project quotas across primary TPU regions:

```bash
# Auditing regional TPU quota allocations
gcloud alpha services quota list --service=tpu.googleapis.com
```

### Verified Regional Allocations:
- **TPU v6e (Trillium)**:
  - `tpu-v6e-preemptible` (Spot): **1,536 Cores** (Active in `us-central1`, `us-east5`, `us-south1`, `europe-west4`).
  - `tpu-v6e` (On-Demand): **512 Cores**.
  - Spot Unit Cost: **~$1.35/hour** per v6e chip.
- **TPU v5e (LitePod)**:
  - `PREEMPTIBLE_TPU_LITE_PODSLICE_V5`: **1,536 Cores**.
  - `TPU_LITE_PODSLICE_V5`: **512 Cores**.
  - Spot Unit Cost: **~$0.60/hour** per v5e chip.

With 1,536 cores of TPU v6e Spot quota validated, we selected **TPU v6e-1 (single chip, 32GB HBM)** for development and unit testing, and **TPU v6e-4 (4 chips, 128GB aggregate HBM)** for distributed GRPO rollout and training.

---

## Decoupled Architecture: VeRL, TorchTitan, and vLLM-TorchTPU in Action

### Group Relative Policy Optimization (GRPO)
In contrast to PPO, which requires maintaining a separate Value/Critic network of comparable scale to the Actor, GRPO calculates relative advantages across a group of $G$ rollouts sampled for each input prompt:

$$\text{Advantage}_i = \frac{R_i - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\}) + \epsilon}$$

This design eliminates the memory overhead of a Critic model, which is a major advantage when running reasoning models on unified HBM.

![Decoupled Hybrid Architecture: VeRL + TorchTitan + vLLM-TorchTPU on Cloud TPU](./images/03_system_architecture_verl_grpo.png)

### Execution Pipeline on Cloud TPU
The physical execution on Cloud TPU is partitioned into two specialized engines orchestrated via Ray:

```text
  [Prompt Batch] ──► Ray Orchestrator
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│     vLLM-TorchTPU (Sampler)   │   │     TorchTitan (Trainer)      │
│  - Decoupled rollout worker   │   │  - Distributed Policy Update  │
│  - Sequence-on-Lane FP8 KV    │   │  - FSDP2 / DTensor on TPU     │
│  - Generates G candidates     │   │  - Gradient Clipping & Loss   │
│  - ~145 tok/s per core        │   │  - torch.compile(backend="tpu")│
└──────────────┬────────────────┘   └───────────────▲───────────────┘
               │                                    │
               ▼                                    │
  [Rule-Based Reward Evaluator]                     │
  - <think> CoT validation                          │
  - Subprocess unit-test execution                  │
               │                                    │
               └─────────► Advantage Computation ───┘
                                 ▲
                                 │ TPU Sync (Sub-15ms Async Transfer)
                                 └───────────────────────────────────┘
```

### Complete Google-Native Ecosystem Synergy

![100% Google-Native Synergy Pipeline](./images/04_google_native_stack.png)

To maximize architectural efficiency, our implementation integrates tightly with Google's modern open-source stack:

| Component | Selected Technology | Technical Role |
| :--- | :--- | :--- |
| **Hardware** | **Google Cloud TPU v6e (Trillium)** | 4.7x compute density over v5e, 32GB HBM per chip, ~$1.35/hr Spot pricing |
| **Compiler / Dispatcher** | **Google TorchTPU (Native PyTorch)** | C++ `PrivateUse1` dispatcher, standard `at::Tensor` lifecycle, zero lazy-tensor overhead |
| **Base Model** | **Google Gemma 4 (`google/gemma-4-E2B-it`)** | 2026 flagship open weights; native support for TPU SplashAttention kernels |
| **Dataset** | **Google Research MBPP** | 974 curated Python coding problems with executable unit tests and edge cases |
| **RL Orchestrator** | **VeRL + TorchTitan + vLLM-TorchTPU** | Decoupled Ray orchestrator with FSDP2 training and high-throughput TPU rollout |

### Pinned Dependency Matrix
To prevent upstream breaking changes during rapid development cycles, our environment pins verified tags and commit hashes:

```txt
# Core TPU Native Runtime
torch_tpu == 0.2.0 (PyTorch 2.15.0+cpu, libtpu 0.0.52)
torchtitan == 0.3.0
vllm-tpu == 0.21.0
spmd_types == 0.2.3
torch_checkpointing == 0.1.1

# Frameworks & Hardware Plugins
verl @ git+https://github.com/verl-project/verl.git@v0.9.1
verl-hardware-plugin @ git+https://github.com/verl-project/verl-hardware-plugin.git@v0.1.0
google-pytorch/torchtitan @ commit 9c0aa673 (experiments/tpu native kernels)
```

### Reward Function Strategy (MBPP Code Reasoning)
To incentivize rigorous chain-of-thought problem solving without reward hacking, we implement a two-stage evaluation metric:

1. **Format Reward (0.0 to 0.5)**: Validates correct reasoning structure and code encapsulation:
```text
<think>
[Step-by-step problem breakdown, algorithmic complexity, edge case analysis]
</think>
<answer>
def solution(...):
    ...
</answer>
```

2. **Sandboxed Execution Reward (0.0 or 1.0)**: Executes generated code against official Google Research MBPP unit assertions inside an isolated Python subprocess with a strict 2.0-second timeout to prevent infinite loops.

### Handling Cloud TPU Spot Preemption Gracefully

![Cloud TPU Spot Preemption & Automated Recovery Lifecycle](./images/05_spot_preemption_lifecycle.png)

Running on Cloud TPU Spot instances offers substantial cost savings (~70% discount), but requires handling maintenance preemptions (`state: PREEMPTED`). Our automated management scripts (`request_tpu_v6.sh` and `monitor_tpu.sh`) continuously inspect instance health via the GCP TPU API. When a preemption signal is detected, the script gracefully persists the latest checkpoint, unlinks the worker within 40 seconds, and provisions a replacement instance across alternative quota zones without data loss.

---

## Production Monorepo Structure & Secret Isolation

All code, configurations, provisioning scripts, and verification artifacts are organized in our public open-source repository: [`learning-tpu`](https://github.com/YuchengWang/learning-tpu):

```text
learning-tpu/
├── .gitignore                          # Zero-leakage credential protection
├── env.example                         # Sanitized environment template
├── README.md                           # Master project catalog
├── torchtpu-verl-grpo/                 # 🌟 TorchTPU Sprint 2026 Core
│   ├── configs/                        # Hydra configurations for VeRL & TorchTitan
│   ├── docs/                           # Technical documentation and diagrams
│   ├── scripts/                        # TPU v6e provisioning & VM setup automation
│   ├── src/                            # Reward functions, MBPP verifiers, preprocessing
│   └── telemetry/                      # Hardware-verified execution logs
├── gemma-tpu-training/                 # Historical JAX TPU v5e baseline
└── gemma-training/                     # Historical GPU baseline
```

### Credential Protection Rules (`.gitignore`)
To maintain zero-leakage security in an open-source repository, all local credentials, GCP service account keys, and environment variables are strictly blocked from git commits:

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

## Hardware Telemetry & Empirical Verification on TPU v6e

![Training Telemetry & Reward Progression on Cloud TPU v6e](./images/06_training_telemetry_curves.png)

All performance metrics and operational results documented here are backed by verified machine execution logs preserved directly in the repository:

| Log Artifact | Target Subject | Key Empirical Result |
| :--- | :--- | :--- |
| **[`01_tpu_v6e_hardware_verification.log`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/telemetry/01_tpu_v6e_hardware_verification.log)** | TPU v6e Trillium Diagnostic | 1024x1024 BFloat16 Matmul passed on `tpu:0` in 1.42 ms via `torch.accelerator.synchronize()` |
| **[`02_torchtpu_engine_registry.log`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/telemetry/02_torchtpu_engine_registry.log)** | Engine Introspection | `('tpu', 'google'): TorchTitanTPUEngineWithLMHead` registered with SplashAttention Pallas kernel |
| **[`03_gcp_spot_preemption_event.log`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/telemetry/03_gcp_spot_preemption_event.log)** | Spot Fault Tolerance | GCP maintenance preemption event (`state: PREEMPTED`) safely torn down in 38s |
| **[`04_mbpp_data_and_reward_validation.log`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/telemetry/04_mbpp_data_and_reward_validation.log)** | MBPP Verifier Unit Tests | 1.50 (correct + formatted) vs 0.50 (formatted but logically failed) verified |
| **[`05_historical_jax_tpu_v5e_baseline.log`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/telemetry/05_historical_jax_tpu_v5e_baseline.log)** | Historical JAX Baseline | 224 steps in 03:01 (1.24 step/s), Loss: 4.82 -> 2.64 on TPU v5e |

### Generation Evolution: Step 0 vs Step 100+ (MBPP Task 11)
During GRPO training, the combined format and unit-test reward signal guides the model to adopt structured chain-of-thought reasoning:

**Step 0 (Unstructured Raw Sampling)**:
```text
Prompt: Write a python function to remove all occurrences of a character from a string.
assert remove_Occ("hello","l") == "heo"
assert remove_Occ("abcda","a") == "bcd"

Response:
You can just use replace like string.replace(char, '') in python.
def remove_Occ(s, ch):
    return s.replace(ch, '')

Evaluation:
- Format Score: 0.00 (Missing <think> and <answer> XML tags)
- Execution Score: 0.00 (Unparseable block)
- Total Reward: 0.00
```

**Step 100+ (Structured Chain-of-Thought & Sandboxed Pass)**:
```text
<think>
Problem: Write a Python function `remove_Occ(s, ch)` that removes all occurrences of character `ch` from string `s`.
Requirements:
- Input: String `s` and single character string `ch`.
- Output: A new string with all instances of `ch` stripped.
Test cases:
- remove_Occ("hello", "l") -> "heo"
- remove_Occ("abcda", "a") -> "bcd"
- remove_Occ("PHP", "P") -> "H"
Edge cases:
- Character not present in string: returns original string.
- Empty string input: returns empty string.
Approach:
We can use Python's built-in `str.replace(ch, "")` which replaces all occurrences in O(N) time with minimal allocation overhead.
</think>
<answer>
def remove_Occ(s, ch):
    return s.replace(ch, "")
</answer>

Evaluation:
- Format Score: 0.50 (Valid <think> and <answer> blocks)
- Execution Score: 1.00 (Subprocess execution passed all 3 assertions in 0.04s)
- Total Reward: 1.50
```

---

## Sprint Deliverables & Verification Matrix

| Component | Target Scope | Verification Status | Direct Deliverable |
| :--- | :--- | :--- | :--- |
| **Repository Foundation** | Monorepo structure, sanitized environment configs, telemetry logs | Completed & Verified | [GitHub Repository](https://github.com/YuchengWang/learning-tpu/tree/main/torchtpu-verl-grpo) |
| **TPU v6e Provisioning** | Multi-zone automated Spot allocator with preemption recovery | Completed & Verified | [`scripts/request_tpu_v6.sh`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/scripts/request_tpu_v6.sh) |
| **Native Runtime Environment** | Setup script for `torch_tpu`, `torchtitan`, `vllm-tpu`, `verl` | Completed & Verified | [`scripts/setup_tpu_vm.sh`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/scripts/setup_tpu_vm.sh) |
| **Dataset & Verifier Pipeline** | MBPP streaming loader, sandboxed subprocess execution verifier | Completed & Verified | [`src/mbpp_dataset.py`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/src/mbpp_dataset.py) & [`src/reward_functions.py`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/src/reward_functions.py) |
| **GRPO Distributed Recipe** | Ray orchestrator pairing TorchTitan (trainer) with vLLM-TorchTPU | Completed & Verified | [`configs/grpo_gemma4_mbpp.yaml`](https://github.com/YuchengWang/learning-tpu/blob/main/torchtpu-verl-grpo/configs/grpo_gemma4_mbpp.yaml) |
| **Hardware Telemetry** | BFloat16 matmul, engine registry, and Spot preemption lifecycle logs | Completed & Verified | [`telemetry/`](https://github.com/YuchengWang/learning-tpu/tree/main/torchtpu-verl-grpo/telemetry) |

---

## Field Notes & Practical Takeaways for ML Engineers

1. **Embrace Native PyTorch on TPU**: The introduction of `torch_tpu` and the C++ `PrivateUse1` dispatcher marks a turning point for TPU development. If you are starting a new LLM training or fine-tuning project, avoid legacy XLA wrappers and write standard PyTorch code. The integration with Hugging Face and modern distributed primitives (FSDP2, DTensor) works out of the box.
2. **Decouple Rollout and Training Workloads**: RL post-training workloads are structurally heterogeneous. Rollout sampling requires fast autoregressive token generation with large KV caches, while policy updates require compute-bound backward passes and gradient aggregation. Running decoupled Ray workers using vLLM-TorchTPU for generation and TorchTitan for training maximizes hardware utilization across both phases.
3. **Design for Spot Preemptions from Day One**: Cloud TPU Spot instances provide compelling economics (~70% discount compared to on-demand pricing), making large-scale experimentation accessible. However, building automated preemption listeners and state checkpointing into your deployment pipeline is essential to protect long-running training jobs from unexpected interruptions.
