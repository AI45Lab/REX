"""Add a local directory to an S3 prefix without replacing any stored object.

Files already stored with equal size+ETag are skipped. If any stored object
differs from its local file, nothing is uploaded and every conflict is listed.
New files are uploaded with ``run.json`` files last and each is verified by
size+ETag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from storage.s3 import AppendOnlyConflictError, CheckpointObjectStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("s3_prefix")
    args = parser.parse_args(argv)
    if not args.local_dir.is_dir():
        parser.error(f"Not a directory: {args.local_dir}")
    store = CheckpointObjectStore.from_env(prefix=args.s3_prefix)
    try:
        entries = store.upload_directory(args.local_dir)
    except AppendOnlyConflictError as error:
        print(f"{args.local_dir} -> {store.prefix}/: {error}", file=sys.stderr)
        return 1
    uploaded = [e for e in entries if e["uploaded"]]
    print(f"{len(entries)} files under {store.prefix}/: {len(uploaded)} uploaded "
          f"({sum(e['size'] for e in uploaded)} bytes), {len(entries) - len(uploaded)} already present; "
          "all verified by size+ETag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
