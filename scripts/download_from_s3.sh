#!/usr/bin/env bash
# Download everything under an object-storage prefix into a local directory.
#
# Usage: bash scripts/download_from_s3.sh S3_PREFIX DEST [--include-glob PATTERN ...] [--exclude-glob PATTERN ...]
#
#   S3_PREFIX       key prefix inside the bucket, e.g. <root>/<model>/<domain>/<run_id>
#   DEST            local directory; each object lands at DEST/<path below S3_PREFIX>
#   --include-glob  only fetch objects whose path below S3_PREFIX matches some
#                   include pattern (repeatable)
#   --exclude-glob  skip objects whose path matches any exclude pattern (repeatable)
#
# Patterns are shell-style globs on the path below S3_PREFIX, where '*' also
# matches '/'. A pattern starting with '*/' also matches at the top level, so
# '*/eval/*' selects eval/ of a node prefix as well as F1/eval/ of a run prefix.
# Examples:
#   a node without weights   --exclude-glob '*.safetensors' --exclude-glob '*.safetensors.index.json'
#   evaluation only          --include-glob '*/eval/*' --include-glob '*/online_eval/*'
#   diagnostics only         --include-glob '*/diagnostics/*'
#   records only             --include-glob '*/run.json'
#   no per-sample traces     --exclude-glob '*/samples*.jsonl'
#
# Files already present with the same size and ETag are skipped. Others are
# written to <file>.part and renamed into place once their size and ETag
# match. Local files that are not in object storage are left untouched.
# Settings come from the environment: REX_S3_BUCKET, REX_S3_ENDPOINT,
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY; values missing there are read from
# the KEY=VALUE file named by REX_S3_CREDENTIALS (parsed, not executed).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -ge 2 ]] || { rex_usage >&2; exit 2; }
S3_PREFIX="$1" DEST="$(realpath -m "$2")"
shift 2
[[ -n "${S3_PREFIX//\//}" ]] || rex_die "S3_PREFIX must not be empty."
FILTERS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-glob|--exclude-glob)
            [[ $# -ge 2 ]] || rex_die "$1 needs a pattern."
            FILTERS+=("$1" "$2"); shift 2 ;;
        *) rex_usage >&2; exit 2 ;;
    esac
done

rex_activate
python -m storage.download "$S3_PREFIX" "$DEST" ${FILTERS[@]+"${FILTERS[@]}"}
