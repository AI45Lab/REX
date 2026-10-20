#!/usr/bin/env bash
# Add a local directory to object storage; stored objects are never replaced.
#
# Usage: bash scripts/upload_to_s3.sh LOCAL_DIR S3_PREFIX
#
#   LOCAL_DIR  directory to upload; every regular file is stored at
#              S3_PREFIX/<path relative to LOCAL_DIR> (symlinks are skipped)
#   S3_PREFIX  key prefix inside the bucket, e.g. <root>/<model>/<domain>/<run_id>/F1
#
# Object storage is append-only. The prefix is listed first and every local
# file is compared with the object at its key:
#   no object                      -> uploaded
#   object with equal size+ETag    -> skipped
#   object that differs            -> conflict
# Any conflict stops the upload before anything is sent, lists every
# conflicting key and exits 1. Stored objects without a local file are left
# alone. run.json files are uploaded after all other files, deeper ones first,
# so a directory's run.json only appears once everything below it is stored.
# Every uploaded file is checked against the stored size and ETag.
# Settings come from the environment: REX_S3_BUCKET, REX_S3_ENDPOINT,
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY; values missing there are read from
# the KEY=VALUE file named by REX_S3_CREDENTIALS (parsed, not executed).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 2 ]] || { rex_usage >&2; exit 2; }
LOCAL_DIR="$(realpath "$1")" S3_PREFIX="$2"
rex_require_dir "$LOCAL_DIR"
[[ -n "${S3_PREFIX//\//}" ]] || rex_die "S3_PREFIX must not be empty."

rex_activate
python -m storage.upload "$LOCAL_DIR" "$S3_PREFIX"
