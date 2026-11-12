#!/bin/bash

# 配置 (v6e Trillium: 下一代性价比首选)
PROJECT_ID="studio-9772401964-cc157"
ZONE="us-central1-a"
ACCELERATOR_TYPE="v6e-1"
RUNTIME_VERSION="v2-alpha-tpuv6e"
NODE_ID="tpu-v6-$(date +%s)"
QUEUED_RESOURCE_ID="qr-v6-$(date +%s)"
MAX_RUN_DURATION="7d"

echo "正在尝试申请最新一代 v6e-1 (Trillium) TPU..."
echo "Project: $PROJECT_ID, Zone: $ZONE"
echo "Estimated Cost: ~$1.35/hour (Spot/Flex-start)"

# 设置项目
gcloud config set project "$PROJECT_ID" --quiet

# 创建资源请求
gcloud alpha compute tpus queued-resources create "$QUEUED_RESOURCE_ID" \
    --node-id "$NODE_ID" \
    --zone "$ZONE" \
    --accelerator-type "$ACCELERATOR_TYPE" \
    --runtime-version "$RUNTIME_VERSION" \
    --provisioning-model=flex-start \
    --max-run-duration="$MAX_RUN_DURATION"

if [ $? -eq 0 ]; then
    echo "✅ 请求已成功提交: $QUEUED_RESOURCE_ID"
    echo "正在等待状态初始化 (30秒)..."
    sleep 35
    STATE=$(gcloud alpha compute tpus queued-resources describe "$QUEUED_RESOURCE_ID" --zone "$ZONE" --format="value(state.state)")
    echo "💎 当前状态: $STATE"
    echo "建议运行以下命令持续监控:"
    echo "./monitor_tpu.sh $QUEUED_RESOURCE_ID $ZONE"
else
    echo "❌ 申请 v6e-1 失败。"
fi
