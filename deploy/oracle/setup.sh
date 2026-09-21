#!/usr/bin/env bash
# Prepare an Ubuntu 22.04 or 24.04 server and start OmniAI Cloud on it.
#
# Run on the server, from the unpacked source, with SITE_ADDRESS set:
#
#   SITE_ADDRESS=203-0-113-9.sslip.io bash deploy/oracle/setup.sh
#
# Safe to run again: every step checks before it changes anything.
set -euo pipefail

: "${SITE_ADDRESS:?set SITE_ADDRESS, for example 203-0-113-9.sslip.io}"
here="$(cd "$(dirname "$0")" && pwd)"

echo "==> Security updates, and automatic ones from now on"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y unattended-upgrades iptables-persistent
sudo dpkg-reconfigure -f noninteractive unattended-upgrades

echo "==> Firewall: SSH, HTTP and HTTPS only"
# Oracle's Ubuntu images reject everything but SSH in iptables, in addition to
# the cloud security list. Open 80 and 443 above that final reject rule.
for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 5 -p tcp -m state --state NEW --dport "$port" -j ACCEPT
  fi
done
sudo netfilter-persistent save

echo "==> Docker, from the Ubuntu archive"
if ! command -v docker >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2
  sudo systemctl enable --now docker
fi
sudo usermod -aG docker "$USER" || true

echo "==> Swap, as a margin while the image builds"
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> Building and starting (the first build downloads about 6 GB of models)"
cd "$here"
sudo SITE_ADDRESS="$SITE_ADDRESS" docker compose up -d --build

echo "==> Waiting for the app to answer"
for _ in $(seq 1 120); do
  if sudo docker compose exec -T omniai curl -fsS http://127.0.0.1:7860/api/health >/dev/null 2>&1; then
    echo "OmniAI Cloud is running at https://$SITE_ADDRESS"
    exit 0
  fi
  sleep 10
done
echo "The app did not become healthy; see: sudo docker compose logs omniai" >&2
exit 1
