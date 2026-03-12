#!/bin/sh
set -eu

mkdir -p /var/run/tailscale /var/cache/tailscale /var/lib/tailscale

tailscaled \
  --tun=userspace-networking \
  --socks5-server=localhost:1080 \
  --outbound-http-proxy-listen=localhost:1080 &

# Give the daemon a moment to come up
sleep 2
echo $TAILSCALE_AUTHKEY

tailscale up \
  --authkey="${TAILSCALE_AUTHKEY}" \
  --hostname="${MODAL_TASK_ID:-modal}"

exec "$@"
