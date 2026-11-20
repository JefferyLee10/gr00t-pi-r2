#!/usr/bin/env bash
# Fetch exactly the upstream trees used by this release. Simulation submodules
# and Git LFS demo assets are unnecessary for the real-robot pipeline.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$root/third_party"

checkout_source() {
    local name=$1 url=$2 commit=$3 dir="$root/third_party/$1"
    if [[ ! -e "$dir" ]]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$url"
    fi
    if [[ ! -d "$dir/.git" || "$(git -C "$dir" remote get-url origin)" != "$url" ]]; then
        echo "Unexpected source directory: $dir" >&2
        return 1
    fi
    if ! git -C "$dir" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$dir" fetch --depth=1 origin "$commit"
        git -C "$dir" checkout --detach --quiet FETCH_HEAD
    fi
    if [[ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
        echo "$dir must be at $commit. Move the existing checkout to keep its changes." >&2
        return 1
    fi
    printf '%s %s\n' "$name" "$commit"
}

apply_if_needed() {
    local dir="$root/third_party/$1" patch="$root/patches/$2"
    if ! git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$dir" apply --check "$patch"
        git -C "$dir" apply "$patch"
    fi
}

groot_url=https://github.com/pi-r2-flow/Isaac-GR00T.git
checkout_source groot_train "$groot_url" 2a39d591a3af24c42bb1d1c9cd708a2dddac0600
checkout_source groot_deploy "$groot_url" 9a00a4739c05b7638931f505fcc7a3c6f1502187
checkout_source pi_r2 https://github.com/pi-r2-flow/pi-r2-flow.git 3af52ca400a6ec7d141416879aa531a6f62f697a
apply_if_needed groot_train absolute-actions.patch
apply_if_needed groot_train local-backbone.patch
apply_if_needed groot_deploy local-backbone.patch
