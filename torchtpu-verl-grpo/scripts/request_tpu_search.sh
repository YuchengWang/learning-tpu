#!/bin/bash

# 配置
PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-}}"
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID not set. Please export PROJECT_ID or set it in .env."
    exit 1
fi
ACCELERATOR_TYPE="v5litepod-1"
RUNTIME_VERSION="v2-tpuv5-lite-cgroup1" # 使用更稳定的版本
MAX_RUN_DURATION="7d"

# 备选区域列表 (按优先级排序)
ZONES=("us-east1-c" "us-east5-a" "us-west4-a" "europe-west4-a" "us-central1-a")

echo "开始多区域尝试申请 TPU (型号: $ACCELERATOR_TYPE)..."
gcloud config set project "$PROJECT_ID" --quiet

for ZONE in "${ZONES[@]}"; do
    TIMESTAMP=$(date +%s)
    NODE_ID="tpu-lite-$TIMESTAMP"
    QR_ID="qr-$TIMESTAMP"

    echo "------------------------------------------------"
    echo "尝试区域: $ZONE"
    echo "Node ID: $NODE_ID, QR ID: $QR_ID"

    # 发起请求
    gcloud alpha compute tpus queued-resources create "$QR_ID" \
        --node-id "$NODE_ID" \
        --zone "$ZONE" \
        --accelerator-type "$ACCELERATOR_TYPE" \
        --runtime-version "$RUNTIME_VERSION" \
        --provisioning-model=flex-start \
        --max-run-duration="$MAX_RUN_DURATION" 2>/tmp/tpu_error.log

    if [ $? -eq 0 ]; then
        echo "✅ 请求已成功提交到 $ZONE 队列。"
        echo "正在等待初步验证状态 (约 30 秒)..."
        
        # 简单等待，看是否会立即转为 FAILED (类似之前的 Internal Error)
        sleep 35
        STATE=$(gcloud alpha compute tpus queued-resources describe "$QR_ID" --zone "$ZONE" --format="value(state.state)")
        
        if [ "$STATE" == "FAILED" ]; then
            ERROR_MSG=$(gcloud alpha compute tpus queued-resources describe "$QR_ID" --zone "$ZONE" --format="value(state.failedData.error.message)")
            echo "❌ $ZONE 请求立即失败: $ERROR_MSG"
            echo "尝试下一个区域..."
            continue
        else
            echo "💎 $ZONE 请求当前状态: $STATE"
            echo "该请求似乎已稳定进入队列。请运行以下命令监控:"
            echo "./monitor_tpu.sh $QR_ID $ZONE"
            exit 0
        fi
    else
        echo "❌ $ZONE 请求发起失败:"
        cat /tmp/tpu_error.log
        echo "尝试下一个区域..."
    fi
done

echo "------------------------------------------------"
echo "所有候选区域均尝试完毕，未能成功建立稳定队列。"
