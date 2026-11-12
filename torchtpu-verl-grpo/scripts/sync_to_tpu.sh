#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEARNING_TPU_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

# Load configuration from local .env if available
if [ -f "$LEARNING_TPU_ROOT/.env" ]; then
    set -a
    source "$LEARNING_TPU_ROOT/.env"
    set +a
elif [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

NODE_ID="${1:-$NODE_ID}"
ZONE="${2:-$ZONE}"
REMOTE_DEST="~/torchtpu-verl-grpo"

if [ -z "$NODE_ID" ] || [ -z "$ZONE" ]; then
    echo "❌ Error: NODE_ID or ZONE not defined."
    echo "Usage: ./sync_to_tpu.sh <NODE_ID> <ZONE>"
    echo "Or set NODE_ID and ZONE in .env"
    exit 1
fi

echo "=========================================================="
echo "📡 Synchronizing Project to Cloud TPU VM"
echo "Target Node: $NODE_ID ($ZONE)"
echo "Remote Path: $REMOTE_DEST"
echo "=========================================================="

# Create remote directory
gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone "$ZONE" --command="mkdir -p $REMOTE_DEST"

# Sync files (excluding git, caches, and large checkpoints)
rsync_exclude="--exclude=.git --exclude=__pycache__ --exclude=outputs --exclude=.venv --exclude=*.pyc"

echo "📤 Uploading project source, configs, and scripts..."
gcloud compute tpus tpu-vm scp --recurse \
    "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/configs" \
    "$PROJECT_ROOT/scripts" \
    "$PROJECT_ROOT/data" \
    "$PROJECT_ROOT/requirements.txt" \
    "$PROJECT_ROOT/README.md" \
    "$NODE_ID:$REMOTE_DEST/" --zone "$ZONE"

# Copy .env securely if exists
if [ -f "$LEARNING_TPU_ROOT/.env" ]; then
    echo "🔐 Syncing environment secrets (.env)..."
    gcloud compute tpus tpu-vm scp "$LEARNING_TPU_ROOT/.env" "$NODE_ID:$REMOTE_DEST/.env" --zone "$ZONE"
fi

echo "=========================================================="
echo "✅ Code sync completed!"
echo "Connect with:"
echo "  gcloud compute tpus tpu-vm ssh $NODE_ID --zone $ZONE"
echo "=========================================================="
