#!/usr/bin/env bash
# Install the pinned Kubernetes tools into ~/.local/bin (Linux x86-64), each
# checked against its release's SHA-256 checksum. With arguments, only those
# tools (CI's deploy-smoke job needs kind, kubectl and kubeconform):
#
#   deploy/install-tools.sh                         # all of them
#   deploy/install-tools.sh kind kubectl kubeconform
set -euo pipefail

KIND=v0.33.0
KUBECTL=v1.37.1        # Kubernetes 1.37, the kind node's version
HELM=v4.3.1
KUBECONFORM=v0.8.0
ARGOCD=v3.5.6          # the version deploy/argocd/install installs

BIN="$HOME/.local/bin"
mkdir -p "$BIN"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"

check() {  # check <file> <expected sha256>
  echo "$2  $1" | sha256sum -c --quiet
}

install_kind() {
  local url=https://github.com/kubernetes-sigs/kind/releases/download/$KIND/kind-linux-amd64
  curl -fsSLo kind "$url"
  check kind "$(curl -fsSL "$url.sha256sum" | cut -d' ' -f1)"
  install -m 0755 kind "$BIN/"
}

install_kubectl() {
  local url=https://dl.k8s.io/release/$KUBECTL/bin/linux/amd64/kubectl
  curl -fsSLo kubectl "$url"
  check kubectl "$(curl -fsSL "$url.sha256")"
  install -m 0755 kubectl "$BIN/"
}

install_helm() {
  local url=https://get.helm.sh/helm-$HELM-linux-amd64.tar.gz
  curl -fsSLo helm.tgz "$url"
  check helm.tgz "$(curl -fsSL "$url.sha256sum" | cut -d' ' -f1)"
  tar -xzf helm.tgz linux-amd64/helm
  install -m 0755 linux-amd64/helm "$BIN/"
}

install_kubeconform() {
  local base=https://github.com/yannh/kubeconform/releases/download/$KUBECONFORM
  curl -fsSLo kubeconform.tgz "$base/kubeconform-linux-amd64.tar.gz"
  check kubeconform.tgz "$(curl -fsSL "$base/CHECKSUMS" \
    | awk '$2 == "kubeconform-linux-amd64.tar.gz" {print $1}')"
  tar -xzf kubeconform.tgz kubeconform
  install -m 0755 kubeconform "$BIN/"
}

install_argocd() {
  local base=https://github.com/argoproj/argo-cd/releases/download/$ARGOCD
  curl -fsSLo argocd "$base/argocd-linux-amd64"
  check argocd "$(curl -fsSL "$base/cli_checksums.txt" \
    | awk '$2 == "argocd-linux-amd64" {print $1}')"
  install -m 0755 argocd "$BIN/"
}

for tool in "${@:-kind kubectl helm kubeconform argocd}"; do
  for name in $tool; do
    "install_$name"
    echo "installed $name"
  done
done
