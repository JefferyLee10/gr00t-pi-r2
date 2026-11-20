#!/usr/bin/env bash
# Apply one historical RGB3 controller diff to a temporary source file and run it.
set -euo pipefail
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -eq 0 || "$1" == "--list" ]]; then
    for file in "$dir"/variants/*.patch; do basename -- "$file" .patch; done
    exit 0
fi
variant=$1
shift
if [[ ! "$variant" =~ ^[a-zA-Z0-9_]+$ || ! -f "$dir/variants/$variant.patch" ]]; then
    echo "Unknown variant '$variant'. Use --list." >&2
    exit 2
fi
work=$(mktemp -d)
trap 'rm -f -- "$work/controller.py"; rmdir -- "$work"' EXIT
patch --silent --fuzz=0 -o "$work/controller.py" "$dir/pir2.py" "$dir/variants/$variant.patch"
PYTHONPATH="$dir${PYTHONPATH:+:$PYTHONPATH}" python "$work/controller.py" "$@"
