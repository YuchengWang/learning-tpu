# 从 GPU 到 TPU：Gemma 3 1B 微调的硬核工程实录与排坑指南

在上一篇文章中，我分享了如何在 16GB 的 T4 GPU 上通过工程 Trick 强行跑通 Gemma 3 的微调。但作为开发者，我们永远在追求更高的算力效率和更低的成本。今天，我决定将战场转移到 **Google Cloud TPU**。

这不仅仅是一次硬件切换，更是一场关于 **JAX/Flax 架构优化、云端资源调度、以及高精度数据提取** 的实战演习。在这篇文章中，我将毫无保留地分享在申请 TPU 资源、解决 Python 版本冲突、以及实现“表格级”精确微调过程中遇到的所有挑战。

---

## 1. 资源之战：避开 Flex-start 的“内部错误”陷阱

在 Google Cloud 上请求 TPU 资源通常有三种模式：按需 (On-Demand)、现货 (Spot) 和 灵活启动 (Flex-start)。

### 遇到的坑：Code 13 内部错误
最初，我参考官方文档尝试使用 `queued-resources` 的 `flex-start` 模式来申请最廉价的 `v5litepod-1` (v5e 单芯片)。然而，在 `us-central1-a` 和 `us-west4-a` 等热门区域，我反复遇到：
> `ERROR: code: 13, message: an internal error has occurred`

**解决办法：**
经过多轮遍历测试，我发现 **单芯片实例 (v5e-1/v6e-1) 对 Flex-start 队列模式的支持并不稳定**。最终，我写了一个自动扫描脚本，遍历了全球 5 个以上的区域，最终在相对冷门的 **`us-south1-a`** 通过 **标准 Spot 模式** 成功抢到了资源。

**工程心得：** 在算力紧缺时，不要在热门区域死磕。通过脚本自动化探测“冷门”区域（如 `us-south1`, `us-east5`）是抢占低价算力的关键。

---

## 2. 环境之战：Python 3.11 与 Flax 的版本博弈

当我们登录到全新的 TPU VM 时，迎接我的是 Ubuntu 22.04 默认的 Python 3.10 环境。

### 遇到的坑：Dependency Hell
当我尝试安装 `flax>=0.11.1` 时，系统报错：
> `ERROR: No matching distribution found for flax>=0.11.1 (Requires-Python >=3.11)`

由于 Gemma 3 采用了最新的架构，强制要求高版本的 Flax。

**解决办法：**
我没有选择简单的降级，而是编写了一个自动化的 `setup_tpu_env.sh` 脚本，在远程 VM 上强行注入 `ppa:deadsnakes/ppa`，安装 **Python 3.11** 及其对应的 `venv`。这确保了我们能使用最前沿的 JAX[TPU] 驱动。

---

## 3. 数据之战：让模型真正“读懂”复杂的经济表格

微调模型不仅仅是输入文本，对于 IMF 这种充满统计数据的报告，如何处理表格是成败的关键。

### 遇到的坑：信息稀释
最初的切片逻辑只是简单的将表格作为段落的背景补充。但在测试中发现，模型很难准确复述出表格中的特定数值（例如日本 2024 年的精确增长率）。

**解决办法：**
我重构了 `process_imf_data.py`，引入了 **“独立表格采样”** 逻辑：
1.  利用 PyMuPDF 将每个 PDF 页面中的表格解析为标准的 Markdown。
2.  **关键创新**：将每个表格作为一个**独立的高质量训练样本**写入 `jsonl`，并配以“请分析 Page X 的宏观经济数据表”的指令。
3.  这样，数据集从 169 个样本增加到了 **224** 个，信息密度显著提升。

---

## 4. 训练之战：忍受 JAX 的“静默五分钟”

在 TPU 上运行 JAX 时，很多新手会以为程序卡死了。

### 技术细节：JIT 编译
当你看到 `--- [EVALUATION] PRE-TRAINING ---` 之后长达几分钟没有输出时，请不要 Ctrl+C。这是 JAX 在针对 TPU 硬件进行 **JIT (Just-In-Time) 编译**。
- **代价**：前 5 分钟没有任何反馈。
- **回报**：一旦编译完成，Gemma 3 1B 的训练速度飙升至 **15.2 steps/sec**。整个 224 个 Step 的训练在 **3 分钟** 内瞬间完成。

---

## 5. 成果验证：从“幻觉”到“专家”

我们设计了一个“硬核数据对比”测试。

**问题**：*Specifically, what are the 2024 GDP growth projections for Japan according to the October 2024 tables?*

- **训练前 (Baseline)**: 回答 “3.3%” (这是一个通用的、错误的预测值)。
- **训练后 (Gemma-Expert)**: 精确回答 **“0.1%”**。
  - **注**：这完全符合 IMF 10 月报告中因汽车供应中断等原因大幅下调后的真实数据。模型成功吸收了表格中的冷门事实。

---

## 6. 最后的工程化：一个自包含的 TPU 训练包

为了让这一过程可复现，我将所有逻辑解耦并封装进了 `gemma-tpu-training` 文件夹：
- **`.env` 管理**: 所有的 Token、项目 ID、学习率均通过环境变量配置。
- **一键同步**: `sync_to_tpu.sh` 实现本地代码与云端的秒级同步。
- **README 指南**: 即使是第一次接触 TPU 的开发者，也能通过 `env.example` 快速上手。

---

## 结语

从 T4 GPU 的“省吃俭用”到 TPU v5e 的“速度与激情”，核心不在于硬件的优劣，而在于如何通过工程化手段解决资源调度、环境依赖和数据结构化的挑战。

本项目的所有源代码现已开源在 GitHub，欢迎大家尝试。

**Source Code**: [GitHub - YuchengWang/OptionAnalyze/gemma-tpu-training](https://github.com/YuchengWang/OptionAnalyze/tree/main/gemma-tpu-training)

**参考文档**: [Google Cloud TPU Documentation](https://docs.cloud.google.com/tpu/docs/intro-to-tpu)
