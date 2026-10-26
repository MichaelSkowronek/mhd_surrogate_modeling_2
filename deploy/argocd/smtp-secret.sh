#!/usr/bin/env bash
# The SMTP password Alertmanager mails the service's alerts with (the
# monitoring Application's alertmanager config): the Secret
# monitoring/alertmanager-smtp, kept out of git and created from a file on
# this machine, a Gmail app password. bootstrap.sh runs this; run it again
# after changing the file (Alertmanager reads the password at every send).
# Without the file the Secret is created empty, so Alertmanager still
# starts, and every email fails (alertmanager_notifications_failed_total)
# until the file exists and this has run again.
#
#   install -D -m 600 /dev/null ~/.config/mhd-surrogate/smtp-password  # then paste the password into it
#   deploy/argocd/smtp-secret.sh
set -euo pipefail

file=${MHD_SMTP_PASSWORD_FILE:-$HOME/.config/mhd-surrogate/smtp-password}

kubectl config use-context kind-mhd >/dev/null
# Argo CD creates the namespace too (CreateNamespace), but may not have yet.
kubectl get namespace monitoring >/dev/null 2>&1 || kubectl create namespace monitoring
if [[ -f $file ]]; then
  password=$file
else
  echo "no $file: alert emails fail until it exists and this runs again" >&2
  password=/dev/null
fi
# Whitespace stripped: Google shows an app password in groups of four.
tr -d '[:space:]' <"$password" \
  | kubectl -n monitoring create secret generic alertmanager-smtp \
    --from-file=password=/dev/stdin --dry-run=client -o yaml \
  | kubectl apply -f -
