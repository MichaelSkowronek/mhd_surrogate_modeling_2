#!/usr/bin/env bash
# Validate every manifest the cluster gets against the Kubernetes 1.37 API
# (kubeconform, strict: unknown fields are errors), the Argo CD
# Applications against their CRD's schema (the CRDs catalog). Argo CD's own
# CRDs are skipped: there is no schema for CustomResourceDefinition objects
# to check them against, and they come verbatim from its release.
set -euo pipefail
cd "$(dirname "$0")/.."

CRDS='https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
kubeconform() {
  command kubeconform -strict -summary -kubernetes-version 1.37.0 \
    -skip CustomResourceDefinition -schema-location default -schema-location "$CRDS" "$@"
}

for kustomization in deploy/k8s/overlays/* deploy/argocd/install; do
  echo "$kustomization:"
  kubectl kustomize "$kustomization" | kubeconform -
done
echo "deploy/argocd (Applications):"
kubeconform deploy/argocd/root.yaml deploy/argocd/apps/*.yaml
