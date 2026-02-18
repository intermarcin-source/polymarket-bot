#!/bin/bash
# ==============================================
# Polymarket Trading Bot - One-Click Deploy
# ==============================================
# Usage: bash deploy/deploy.sh <server-ip> [ssh-user]
#
# Example:
#   bash deploy/deploy.sh 123.45.67.89
#   bash deploy/deploy.sh 123.45.67.89 root
# ==============================================

set -e

SERVER_IP="${1:?Usage: bash deploy/deploy.sh <server-ip> [ssh-user]}"
SSH_USER="${2:-root}"
REMOTE_DIR="/opt/polymarket-bot"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  Deploying Polymarket Bot"
echo "  Server: ${SSH_USER}@${SERVER_IP}"
echo "  Local:  ${LOCAL_DIR}"
echo "============================================"

# ── 1. Upload Code ──
echo ""
echo "[1/4] Uploading code to server..."
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'data/*.json' \
    --exclude 'logs/' \
    --exclude '.env' \
    --exclude 'dashboard.html' \
    --exclude '.git/' \
    "${LOCAL_DIR}/" "${SSH_USER}@${SERVER_IP}:${REMOTE_DIR}/"

echo "  Code uploaded successfully"

# ── 2. Copy .env.example as .env if .env doesn't exist ──
echo ""
echo "[2/4] Checking .env on server..."
ssh "${SSH_USER}@${SERVER_IP}" "
    if [ ! -f ${REMOTE_DIR}/.env ]; then
        cp ${REMOTE_DIR}/.env.example ${REMOTE_DIR}/.env
        echo '  Created .env from template - EDIT IT BEFORE STARTING!'
    else
        echo '  .env already exists, keeping current config'
    fi
"

# ── 3. Run Server Setup ──
echo ""
echo "[3/4] Running server setup..."
ssh "${SSH_USER}@${SERVER_IP}" "bash ${REMOTE_DIR}/deploy/server-setup.sh"

# ── 4. Start Services ──
echo ""
echo "[4/4] Starting services..."
ssh "${SSH_USER}@${SERVER_IP}" "
    systemctl start polymarket-bot
    systemctl start polymarket-dashboard
    systemctl start polymarket-dashboard-gen.timer
    echo 'All services started!'
    echo ''
    systemctl status polymarket-bot --no-pager -l
"

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "  Dashboard: http://${SERVER_IP}/"
echo "  Bot logs:  ssh ${SSH_USER}@${SERVER_IP} 'journalctl -u polymarket-bot -f'"
echo "  Status:    ssh ${SSH_USER}@${SERVER_IP} 'systemctl status polymarket-bot'"
echo ""
echo "  IMPORTANT: Edit .env on the server before live trading!"
echo "  ssh ${SSH_USER}@${SERVER_IP} 'nano ${REMOTE_DIR}/.env'"
echo ""
echo "============================================"
