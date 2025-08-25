#!/usr/bin/env bash
set -euo pipefail

REMOTE="root@172.237.119.193"
DEST="/root/btc-llm-trading"

# 1) ensure target dir
ssh -o StrictHostKeyChecking=no "$REMOTE" "mkdir -p $DEST"

# 2) sync source only
rsync -az --delete --partial \
--exclude-from=.deployignore \
./ "$REMOTE:$DEST/"

# 3) build & run on server
ssh "$REMOTE" "cd $DEST && docker compose up -d --build && docker system prune -f"

# 4) quick health check
ssh "$REMOTE" "curl -s http://localhost/readyz || true"
