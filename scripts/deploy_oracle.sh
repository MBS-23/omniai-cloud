#!/usr/bin/env bash
# Deploy the committed code to an Ubuntu server over SSH.
#
#   scripts/deploy_oracle.sh 203.0.113.9 ~/.ssh/omniai_oracle
#
# Sends a tarball of HEAD (never the working tree, so nothing uncommitted and
# no .env file can travel), unpacks it on the server and runs
# deploy/oracle/setup.sh there. The site is served at <ip-with-dashes>.sslip.io,
# a public DNS name that resolves to the IP, so HTTPS works without a domain.
set -euo pipefail

host="${1:?usage: deploy_oracle.sh <server-ip> [ssh-key]}"
key="${2:-$HOME/.ssh/omniai_oracle}"
user="${DEPLOY_USER:-ubuntu}"
site="${SITE_ADDRESS:-$(echo "$host" | tr . -).sslip.io}"
ssh_opts=(-i "$key" -o StrictHostKeyChecking=accept-new)

root="$(cd "$(dirname "$0")/.." && pwd)"
archive="$(mktemp -t omniai.XXXXXX).tar.gz"
trap 'rm -f "$archive"' EXIT

git -C "$root" archive --format=tar.gz -o "$archive" HEAD
scp "${ssh_opts[@]}" "$archive" "$user@$host:omniai.tar.gz"
ssh "${ssh_opts[@]}" "$user@$host" "
  set -e
  sudo mkdir -p /opt/omniai
  sudo tar -xzf ~/omniai.tar.gz -C /opt/omniai
  rm ~/omniai.tar.gz
  SITE_ADDRESS='$site' bash /opt/omniai/deploy/oracle/setup.sh
"
