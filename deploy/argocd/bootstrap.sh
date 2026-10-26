#!/usr/bin/env bash
# Install Argo CD on the kind cluster (deploy/kind/create-cluster.sh) and
# hand it the root Application; from then on the cluster follows main.
set -euo pipefail
cd "$(dirname "$0")/../.."

kubectl config use-context kind-mhd
# Server-side: Argo CD's CRDs are too large for client-side apply.
kubectl apply --server-side -k deploy/argocd/install
for workload in deployment/argocd-server deployment/argocd-repo-server \
  statefulset/argocd-application-controller; do
  kubectl -n argocd rollout status "$workload" --timeout=300s
done
# The one Secret not in git: Alertmanager's SMTP password.
deploy/argocd/smtp-secret.sh
kubectl apply -f deploy/argocd/root.yaml

echo "Argo CD: http://localhost:8081, user admin, password:"
echo "  kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d"
