#!/bin/bash
set -e

# ==============================================================================
# Google Cloud TPU v6e (Trillium) Automated Provisioning Script
# Prioritizes direct Spot allocation with multi-zone fallback
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARNING_TPU_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load configuration from .env if available
if [ -f "$LEARNING_TPU_ROOT/.env" ]; then
    set -a
    source "$LEARNING_TPU_ROOT/.env"
    set +a
fi

PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-studio-9772401964-cc157}}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v6e-1}"
VERSION="${VERSION:-v2-alpha-tpuv6e}"
NODE_ID="${NODE_ID:-tpu-v6-$(date +%s)}"

# High-availability zones with confirmed v6e quota and accelerator type
PREFERRED_ZONES=("us-central1-a" "us-east4-a" "us-east5-c" "us-south1-a" "europe-west4-a")

echo "=========================================================="
echo "🚀 Requesting TPU v6e ($ACCELERATOR_TYPE) - Spot Instance"
echo "Project: $PROJECT_ID"
echo "Node ID: $NODE_ID"
echo "Price Estimate: ~$1.35/hour (Spot)"
echo "=========================================================="

gcloud config set project "$PROJECT_ID" --quiet

SUCCESS=0
for ZONE in "${PREFERRED_ZONES[@]}"; do
    echo "🔎 Attempting direct Spot creation in zone: $ZONE..."
    if gcloud compute tpus tpu-vm create "$NODE_ID" \
        --zone="$ZONE" \
        --accelerator-type="$ACCELERATOR_TYPE" \
        --version="$VERSION" \
        --spot \
        --quiet; then
        
        echo ""
        echo "🎉 SUCCESS: TPU v6e Spot instance created!"
        echo "Node ID: $NODE_ID"
        echo "Zone:    $ZONE"
        echo ""
        echo "Update your .env with:"
        echo "NODE_ID=\"$NODE_ID\""
        echo "ZONE=\"$ZONE\""
        echo ""
        echo "Connect via SSH:"
        echo "gcloud compute tpus tpu-vm ssh $NODE_ID --zone $ZONE"
        SUCCESS=1
        break
    else
        echo "⚠️ Capacity unavailable in $ZONE, trying next zone..."
    fi
done

if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "❌ Spot capacity temporarily exhausted in standard zones."
    echo "Fallback: Submitting Queued Resource request in ${PREFERRED_ZONES[0]}..."
    
    QR_ID="qr-v6-$(date +%s)"
    gcloud alpha compute tpus queued-resources create "$QR_ID" \
        --node-id="$NODE_ID" \
        --zone="${PREFERRED_ZONES[0]}" \
        --accelerator-type="$ACCELERATOR_TYPE" \
        --runtime-version="$VERSION" \
        --provisioning-model=flex-start \
        --max-run-duration="7d"
        
    echo "✅ Queued Resource request submitted: $QR_ID"
    echo "Monitor with: ./scripts/monitor_tpu.sh $QR_ID ${PREFERRED_ZONES[0]}"
fi
