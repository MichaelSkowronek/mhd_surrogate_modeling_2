#!/usr/bin/env bash
# Create the local kind cluster (deploy/kind/cluster.yaml).
#
# kind runs its node in a Docker network called `kind`, which it creates
# with a subnet Docker picks, so the host's address as the cluster sees it
# (the network's gateway) would differ from machine to machine. The pods
# reach the tracking server on the host there (deploy/k8s/overlays/kind's
# hostAliases), so this creates the network first, with a fixed subnet;
# kind uses an existing `kind` network as it is.
set -euo pipefail
cd "$(dirname "$0")/../.."

NETWORK=kind
SUBNET=172.30.0.0/16
GATEWAY=172.30.0.1
CLUSTER=mhd

if ! docker network inspect "$NETWORK" > /dev/null 2>&1; then
  # The options kind gives the network it creates itself.
  docker network create "$NETWORK" --driver bridge --subnet "$SUBNET" --gateway "$GATEWAY" \
    -o com.docker.network.bridge.enable_ip_masquerade=true \
    -o com.docker.network.driver.mtu=1500
fi
gateway=$(docker network inspect "$NETWORK" -f '{{range .IPAM.Config}}{{.Gateway}} {{end}}')
if [ "$gateway" != "$GATEWAY " ]; then
  echo "the $NETWORK network's gateway is '$gateway', not $GATEWAY: delete its kind" \
    "clusters and the network (docker network rm $NETWORK), then run this again" >&2
  exit 1
fi

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --config deploy/kind/cluster.yaml
fi
kubectl cluster-info --context "kind-$CLUSTER"
