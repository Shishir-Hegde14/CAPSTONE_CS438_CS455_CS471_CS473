#!/usr/bin/env bash
# Stage a user-owned copy of /usr for unprivileged bind mounts in WSL.
# No sudo and no system files are changed; this may use several GB of WSL disk.
set -euo pipefail

HOST_USR="${OS_SANDBOX_HOST_USR:-$HOME/.cache/lab_8b_kit/usr}"
PARENT="$(dirname "$HOST_USR")"
mkdir -p "$PARENT"

if [[ -d "$HOST_USR/bin" && -d "$HOST_USR/lib" && -d "$HOST_USR/sbin" ]]; then
  echo "Existing /usr snapshot looks complete: $HOST_USR"
  du -sh "$HOST_USR"
  exit 0
fi
if [[ -e "$HOST_USR" ]]; then
  echo "Found an incomplete snapshot at $HOST_USR; inspect it before removing it." >&2
  exit 2
fi

# WSL can expose an unreadable ext4 lost+found under its kernel modules tree.
# It is filesystem-recovery metadata, not a runtime binary/library; exclude it.
required_kb="$(du -sk --exclude='*/lost+found' /usr 2>/dev/null | awk '{print $1}')"
available_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
if [[ -z "$required_kb" || -z "$available_kb" || "$available_kb" -lt "$required_kb" ]]; then
  echo "Not enough available WSL disk space for a copy of /usr." >&2
  echo "Needed at least ${required_kb:-unknown} KiB; available ${available_kb:-unknown} KiB." >&2
  exit 2
fi

tmp="${HOST_USR}.stage.$$"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM
mkdir -p "$tmp"
echo "Copying /usr into a user-owned WSL directory (no sudo); this can take a few minutes."
# tar preserves directory structure, symlinks and executable bits. Files are
# created as this user, making the tree bind-mountable in a rootless userns.
# The single inaccessible WSL lost+found directory is deliberately excluded.
(cd /usr && tar --exclude='./lib/modules/*/lost+found' -cf - .) \
  | tar -C "$tmp" -xf - --no-same-owner
for d in bin lib sbin; do
  [[ -d "$tmp/$d" ]] || { echo "Copy is missing $d" >&2; exit 1; }
done
mv "$tmp" "$HOST_USR"
trap - EXIT INT TERM
echo "Snapshot ready: $HOST_USR"
du -sh "$HOST_USR"
echo "The evaluator will mount this snapshot read-only inside each task namespace."
