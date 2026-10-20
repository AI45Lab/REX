"""Download every object under an S3 prefix into a local directory.

Objects already present locally with equal size+ETag are skipped; others are
written to ``<file>.part`` and renamed into place once their size+ETag match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from storage.s3 import CheckpointObjectStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s3_prefix")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--include-glob", action="append", default=[], metavar="PATTERN",
                        help="only fetch objects whose path below the prefix matches (repeatable)")
    parser.add_argument("--exclude-glob", action="append", default=[], metavar="PATTERN",
                        help="skip objects whose path below the prefix matches (repeatable)")
    args = parser.parse_args(argv)
    store = CheckpointObjectStore.from_env(prefix=args.s3_prefix)
    summary = store.download_tree(args.dest, include=args.include_glob, exclude=args.exclude_glob)
    print(f"{summary['objects']} objects under {store.prefix}/ -> {args.dest}: "
          f"{summary['downloaded']} downloaded, {summary['present']} already present, "
          f"{summary['filtered_out']} filtered out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
