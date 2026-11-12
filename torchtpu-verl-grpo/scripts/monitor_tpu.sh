#!/bin/bash

# 配置信息（默认值，也可以通过参数传入）
QR_ID="${1:-qr-1774921847}"
ZONE="${2:-us-central1-a}"

echo "开始监控 TPU 队列资源: $QR_ID (区域: $ZONE)"
echo "当前时间: $(date)"
echo "------------------------------------------------"

while true; do
    # 获取当前状态
    # 状态可能为: WAITING_FOR_RESOURCES, PROVISIONING, ACTIVE, FAILED 等
    STATUS=$(gcloud alpha compute tpus queued-resources describe "$QR_ID" --zone "$ZONE" --format="value(state.state)" 2>/dev/null)
    
    if [ -z "$STATUS" ]; then
        echo "错误: 无法获取资源状态，请检查 QR_ID 是否正确。"
        exit 1
    fi

    echo "[$(date +%H:%M:%S)] 当前状态: $STATUS"

    # 如果状态是 ACTIVE，表示资源已就绪
    if [ "$STATUS" == "ACTIVE" ]; then
        echo -e "\n\a🚀 恭喜！TPU 资源已就绪 (ACTIVE)！"
        echo "你可以现在连接到 TPU 节点了。"
        
        # 获取 Node ID (通常与 QR_ID 相关联，或者从描述中提取)
        NODE_ID=$(gcloud alpha compute tpus queued-resources describe "$QR_ID" --zone "$ZONE" --format="value(tpu.nodeSpec[0].nodeId)")
        echo "Node ID: $NODE_ID"
        echo "连接命令示例: gcloud compute tpus tpu-vm ssh $NODE_ID --zone $ZONE"
        
        # 退出监控
        break
    fi

    # 如果状态是 FAILED，表示请求失败
    if [ "$STATUS" == "FAILED" ]; then
        echo -e "\n❌ 请求失败。请检查控制台或使用 describe 命令查看详细错误原因。"
        exit 1
    fi

    # 每 30 秒检查一次
    sleep 30
done
