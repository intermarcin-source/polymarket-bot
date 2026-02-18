#!/bin/bash
# ==============================================
# Polymarket Trading Bot - Server Setup Script
# ==============================================
# Run as root on a fresh Hetzner Ubuntu server:
#   bash server-setup.sh
# ==============================================

set -e

echo "============================================"
echo "  Polymarket Bot - Server Setup"
echo "============================================"

# ── 1. System Update ──
echo "[1/8] Updating system packages..."
apt update && apt upgrade -y

# ── 2. Install Dependencies ──
echo "[2/8] Installing Python 3, pip, venv, git, nginx, ufw..."
apt install -y python3 python3-pip python3-venv python3-full git nginx ufw

# ── 3. Create Bot User ──
echo "[3/8] Creating botuser..."
if id "botuser" &>/dev/null; then
    echo "  botuser already exists, skipping"
else
    useradd -m -s /bin/bash botuser
    echo "  Created botuser"
fi

# ── 4. Create Project Directory ──
echo "[4/8] Setting up /opt/polymarket-bot..."
mkdir -p /opt/polymarket-bot
# Remove any root-owned venv leftover
rm -rf /opt/polymarket-bot/venv
chown -R botuser:botuser /opt/polymarket-bot

# ── 5. Setup Python Virtual Environment ──
echo "[5/8] Creating Python virtual environment..."
su - botuser -c "
  cd /opt/polymarket-bot
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
"

# ── 6. Create Directories ──
echo "[6/8] Creating data and logs directories..."
su - botuser -c "
  mkdir -p /opt/polymarket-bot/data
  mkdir -p /opt/polymarket-bot/logs
"

# ── 7. Install Systemd Services ──
echo "[7/8] Installing systemd services..."
cp /opt/polymarket-bot/deploy/polymarket-bot.service /etc/systemd/system/
cp /opt/polymarket-bot/deploy/polymarket-dashboard.service /etc/systemd/system/
cp /opt/polymarket-bot/deploy/polymarket-dashboard-gen.service /etc/systemd/system/
cp /opt/polymarket-bot/deploy/polymarket-dashboard-gen.timer /etc/systemd/system/

systemctl daemon-reload
systemctl enable polymarket-bot.service
systemctl enable polymarket-dashboard.service
systemctl enable polymarket-dashboard-gen.timer

# ── 8. Setup Nginx ──
echo "[8/8] Configuring nginx..."
cp /opt/polymarket-bot/deploy/nginx-polymarket.conf /etc/nginx/sites-available/polymarket
ln -sf /etc/nginx/sites-available/polymarket /etc/nginx/sites-enabled/polymarket
rm -f /etc/nginx/sites-enabled/default

# Create HTTP basic auth using Python (no apache2-utils needed)
echo ""
echo "Creating dashboard login credentials..."
echo "Enter a password for the dashboard (username: admin):"
read -s DASH_PASS
HASH=$(python3 -c "import crypt; print(crypt.crypt('$DASH_PASS', crypt.mksalt(crypt.METHOD_SHA256)))")
echo "admin:$HASH" > /etc/nginx/.htpasswd
echo "  Dashboard credentials saved."

nginx -t && systemctl restart nginx

# ── 9. Setup Firewall ──
echo "Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
echo "y" | ufw enable

# ── 10. Setup Log Rotation ──
cp /opt/polymarket-bot/deploy/logrotate-polymarket /etc/logrotate.d/polymarket-bot

# Add cron job to clean old Python log files
su - botuser -c '(crontab -l 2>/dev/null; echo "0 3 * * * find /opt/polymarket-bot/logs -name \"bot_*.log\" -mtime +14 -delete") | crontab -'

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo "  1. Edit the .env file:"
echo "     nano /opt/polymarket-bot/.env"
echo ""
echo "  2. Start the bot:"
echo "     systemctl start polymarket-bot"
echo "     systemctl start polymarket-dashboard"
echo "     systemctl start polymarket-dashboard-gen.timer"
echo ""
echo "  3. Check status:"
echo "     systemctl status polymarket-bot"
echo "     journalctl -u polymarket-bot -f"
echo ""
echo "  4. View dashboard:"
echo "     http://YOUR_SERVER_IP/"
echo "     Login: admin / (password you just set)"
echo ""
echo "============================================"
