"""Copy directory trees to and from S3-compatible object storage.

Verification cannot rely on the endpoint's checksum headers: the deployment in
use accepts a deliberately wrong ``ChecksumSHA256`` without complaint. Single
part uploads do return an ETag equal to the MD5 of the body, and multipart
uploads return a deterministic ``<md5-of-part-digests>-<n>`` form, so this
module recomputes both locally and compares them, together with the size.
That size+ETag comparison is the only integrity check.

Storage is append-only: an upload never replaces a stored object. Files that
are already stored with equal size+ETag are skipped; any stored object that
differs stops the upload before anything is sent.

Settings come from the environment: ``REX_S3_BUCKET``, ``REX_S3_ENDPOINT``,
``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``. Missing values may be
filled from a ``KEY=VALUE`` file named by ``REX_S3_CREDENTIALS``; that file is
parsed, never executed.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from common.manifest_names import RUN_RECORD

# Chosen so a ~1.5 GiB shard uploads in a bounded number of parts while staying
# well under the 10k-part limit for the largest checkpoints seen here.
MULTIPART_CHUNK_BYTES = 64 * 1024 * 1024
MULTIPART_THRESHOLD_BYTES = 64 * 1024 * 1024
MAX_CONCURRENCY = 8
_READ_CHUNK = 8 * 1024 * 1024
_SETTINGS = ("REX_S3_BUCKET", "REX_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


class ObjectStoreUnavailableError(RuntimeError):
    """Raised when object storage is requested but cannot be configured."""


class AppendOnlyConflictError(RuntimeError):
    """Raised when an upload would replace a stored object with different content."""

    def __init__(self, conflicts: list[dict[str, Any]]) -> None:
        self.conflicts = conflicts
        lines = [f"  {c['key']}: local {c['local_size']} bytes, stored {c['remote_size']} bytes "
                 f"(ETag {c['remote_etag']})" for c in conflicts]
        super().__init__(f"{len(conflicts)} stored object(s) differ from the local files; "
                         "object storage is append-only, nothing was uploaded:\n" + "\n".join(lines))


def expected_etag(
    path: Path,
    *,
    chunk_bytes: int = MULTIPART_CHUNK_BYTES,
    threshold_bytes: int = MULTIPART_THRESHOLD_BYTES,
) -> str:
    """Reproduce the ETag S3 reports for a body uploaded from ``path``.

    Small objects yield the MD5 hex digest. Objects sent as multipart yield the
    MD5 of the concatenated part digests, suffixed with the part count.
    """
    if chunk_bytes <= 0 or threshold_bytes <= 0:
        raise ValueError("Multipart sizes must be positive.")
    size = path.stat().st_size
    if size < threshold_bytes:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_READ_CHUNK), b""):
                digest.update(block)
        return digest.hexdigest()

    part_digests: list[bytes] = []
    with path.open("rb") as handle:
        while True:
            part = handle.read(chunk_bytes)
            if not part:
                break
            part_digests.append(hashlib.md5(part).digest())
    combined = hashlib.md5(b"".join(part_digests)).hexdigest()
    return f"{combined}-{len(part_digests)}"


def read_credentials_file(path: Path) -> dict[str, str]:
    """Read a KEY=VALUE secrets file without shell evaluation or credential output."""
    settings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if separator:
            settings[key.strip()] = value.strip().strip("'\"")
    return settings


def _not_found(error: Exception) -> bool:
    code = getattr(error, "response", {}).get("Error", {}).get("Code")
    return code in {"404", "NoSuchKey", "NotFound"}


def _glob_matches(relative: str, pattern: str) -> bool:
    # '*' also matches '/'. Trying the path with a leading '/' as well lets a
    # pattern like '*/eval/*' or '*/run.json' cover the top level of the
    # prefix too, so the same pattern works for a run prefix and a node prefix.
    return fnmatch.fnmatchcase(relative, pattern) or fnmatch.fnmatchcase("/" + relative, pattern)


def _matches_globs(relative: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    include, exclude = list(include), list(exclude)
    if include and not any(_glob_matches(relative, pattern) for pattern in include):
        return False
    return not any(_glob_matches(relative, pattern) for pattern in exclude)


def _upload_order(relative: str) -> tuple:
    # run.json after every other file, deeper ones first: each record then
    # follows everything stored below its directory.
    parts = PurePosixPath(relative).parts
    if parts[-1] == RUN_RECORD:
        return (1, -len(parts), relative)
    return (0, 0, relative)


class CheckpointObjectStore:
    """Upload and download directory trees under one bucket prefix."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore.config import Config
        except ImportError as error:  # pragma: no cover - depends on image
            raise ObjectStoreUnavailableError(
                "boto3 is required to use object storage"
            ) from error

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            # The endpoint is path style; virtual-host addressing would send
            # the bucket into a hostname it does not serve.
            config=Config(s3={"addressing_style": "path"},
                          max_pool_connections=2 * MAX_CONCURRENCY),
        )
        self._transfer = TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD_BYTES,
            multipart_chunksize=MULTIPART_CHUNK_BYTES,
            max_concurrency=MAX_CONCURRENCY,
            use_threads=True,
        )

    @classmethod
    def from_env(cls, *, prefix: str) -> "CheckpointObjectStore":
        """Build from the environment, filling gaps from ``REX_S3_CREDENTIALS``.

        Credentials stay in the process environment or the operator's secrets
        file: writing them next to the data would place secrets on shared
        storage.
        """
        settings: dict[str, str] = {}
        credentials = os.environ.get("REX_S3_CREDENTIALS")
        if credentials:
            path = Path(credentials)
            if not path.exists():
                raise ObjectStoreUnavailableError(f"REX_S3_CREDENTIALS does not exist: {path}")
            settings.update(read_credentials_file(path))
        settings.update({name: os.environ[name] for name in _SETTINGS if os.environ.get(name)})
        missing = [name for name in _SETTINGS if not settings.get(name)]
        if missing:
            raise ObjectStoreUnavailableError(
                "Missing object-store settings: " + ", ".join(missing)
            )
        return cls(
            bucket=settings["REX_S3_BUCKET"],
            prefix=prefix,
            endpoint_url=settings["REX_S3_ENDPOINT"],
            access_key=settings["AWS_ACCESS_KEY_ID"],
            secret_key=settings["AWS_SECRET_ACCESS_KEY"],
        )

    def key_for(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def relative_for(self, key: str) -> str:
        """Path of ``key`` below the prefix, refusing anything that could escape it."""
        relative = key[len(self.prefix) + 1:] if self.prefix else key
        path = PurePosixPath(relative)
        if (self.prefix and not key.startswith(self.prefix + "/")) or not relative \
                or path.is_absolute() or ".." in path.parts or str(path) != relative:
            raise ValueError(f"Unsafe object key below {self.prefix!r}: {key}")
        return relative

    # ---- size + ETag -------------------------------------------------------
    def _head(self, key: str) -> tuple[int, str] | None:
        try:
            head = self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            if _not_found(error):
                return None
            raise
        return head["ContentLength"], head["ETag"].strip('"')

    def _local_etag(self, path: Path, key: str, remote_etag: str) -> str:
        """Hash ``path`` the way the stored object was hashed.

        A plain ETag is one MD5 even when the file exceeds the part size. A
        multipart ETag depends on the uploader's part size, which objects
        written by other tools need not share with ours, so it is read from
        the first part of the stored object.
        """
        if "-" not in remote_etag:
            return expected_etag(path, threshold_bytes=path.stat().st_size + 1)
        chunk = MULTIPART_CHUNK_BYTES
        try:
            part = self._client.head_object(Bucket=self.bucket, Key=key, PartNumber=1)
            chunk = int(part["ContentLength"]) or chunk
        except Exception:
            pass  # Endpoints without part HEAD support fall back to our own part size.
        return expected_etag(path, chunk_bytes=chunk, threshold_bytes=1)

    def matches(self, path: Path, key: str) -> bool:
        """Whether the object at ``key`` exists with the size and ETag of ``path``."""
        remote = self._head(key)
        if remote is None or remote[0] != path.stat().st_size:
            return False
        return self._local_etag(path, key, remote[1]) == remote[1]

    def verify_object(self, path: Path, key: str) -> dict[str, Any]:
        remote = self._head(key)
        if remote is None:
            raise RuntimeError(f"Object missing after transfer: {key}")
        remote_size, remote_etag = remote
        local_size = path.stat().st_size
        if remote_size != local_size:
            raise RuntimeError(
                f"Size mismatch for {key}: local {local_size}, remote {remote_size}"
            )
        local_etag = self._local_etag(path, key, remote_etag)
        if remote_etag != local_etag:
            raise RuntimeError(
                f"ETag mismatch for {key}: local {local_etag}, remote {remote_etag}"
            )
        return {"key": key, "size": local_size, "etag": remote_etag}

    # ---- upload ------------------------------------------------------------
    def upload_verified(self, path: Path, relative: str) -> dict[str, Any]:
        """Upload one file and confirm the stored object matches it."""
        key = self.key_for(relative)
        self._client.upload_file(str(path), self.bucket, key, Config=self._transfer)
        return self.verify_object(path, key)

    def plan_upload(self, directory: Path, relative_root: str = "") -> dict[str, list]:
        """Compare every regular file under ``directory`` with what is stored.

        Returns ``new`` (no object at the key) as ``(path, relative)`` pairs,
        ``present`` (equal size and ETag) as ``(path, relative, etag)``, and
        ``conflicts``: stored objects that differ from the local file.
        Symlinks are skipped: object storage has no such concept, and the
        milestone links this project writes are local navigation aids that
        the directory contents already describe.
        """
        stored = {entry["key"]: entry for entry in self.list_prefix()}
        plan: dict[str, list] = {"new": [], "present": [], "conflicts": []}
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            if relative_root:
                relative = f"{relative_root.strip('/')}/{relative}"
            key = self.key_for(relative)
            remote = stored.get(key)
            if remote is None:
                plan["new"].append((path, relative))
                continue
            size = path.stat().st_size
            if size == remote["size"] and self._local_etag(path, key, remote["etag"]) == remote["etag"]:
                plan["present"].append((path, relative, remote["etag"]))
            else:
                plan["conflicts"].append({"key": key, "local_size": size, "remote_size": remote["size"],
                                          "remote_etag": remote["etag"]})
        return plan

    def upload_directory(self, directory: Path, relative_root: str = "") -> list[dict[str, Any]]:
        """Add every regular file under ``directory`` without replacing anything.

        Storage is append-only: files already stored with equal size+ETag are
        skipped, and if any stored object differs from its local file nothing
        is uploaded and :class:`AppendOnlyConflictError` lists them all. New
        files are uploaded with ``run.json`` files last, deepest first, so a
        directory's record only appears once everything below it is stored.
        Each upload is verified by size+ETag. Each returned entry has the
        object's key, size, ETag and whether it was transferred (``uploaded``).
        """
        plan = self.plan_upload(directory, relative_root)
        if plan["conflicts"]:
            raise AppendOnlyConflictError(plan["conflicts"])
        results: list[dict[str, Any]] = []
        for path, relative, etag in plan["present"]:
            results.append({"key": self.key_for(relative), "size": path.stat().st_size,
                            "etag": etag, "uploaded": False})
            print(f"present  {relative}", flush=True)
        for path, relative in sorted(plan["new"], key=lambda item: _upload_order(item[1])):
            key = self.key_for(relative)
            # Narrow the window in which another writer could have stored the
            # key since the listing (no conditional put is relied on).
            remote = self._head(key)
            if remote is not None:
                if not self.matches(path, key):
                    raise AppendOnlyConflictError([{"key": key, "local_size": path.stat().st_size,
                                                    "remote_size": remote[0], "remote_etag": remote[1]}])
                results.append({"key": key, "size": remote[0], "etag": remote[1], "uploaded": False})
                print(f"present  {relative}", flush=True)
                continue
            results.append(dict(self.upload_verified(path, relative), uploaded=True))
            print(f"uploaded {relative}", flush=True)
        return results

    # ---- listing and download ---------------------------------------------
    def list_prefix(self) -> list[dict[str, Any]]:
        """Every object below the prefix, as key, size and ETag."""
        entries = []
        token = None
        listing_prefix = self.prefix + "/" if self.prefix else ""
        while True:
            page = self._client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=listing_prefix,
                **({"ContinuationToken": token} if token else {}),
            )
            for obj in page.get("Contents", []):
                entries.append(
                    {"key": obj["Key"], "size": obj["Size"], "etag": obj["ETag"].strip('"')}
                )
            if not page.get("IsTruncated"):
                return sorted(entries, key=lambda item: item["key"])
            token = page["NextContinuationToken"]

    def download_verified(self, key: str, target: Path) -> bool:
        """Fetch ``key`` into ``target`` unless an equal file is already there.

        The body is written to ``<target>.part`` and only renamed into place
        after it matches the stored size and ETag. Returns whether a transfer
        happened.
        """
        if target.is_file() and self.matches(target, key):
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        try:
            self._client.download_file(self.bucket, key, str(temporary), Config=self._transfer)
            self.verify_object(temporary, key)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def download_tree(
        self,
        destination: Path,
        *,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
    ) -> dict[str, int]:
        """Download every object below the prefix into ``destination/<relative>``.

        ``include``/``exclude`` are glob patterns matched against the relative
        path; an object is fetched when it matches some include pattern (or
        none are given) and no exclude pattern.
        """
        root = destination.resolve()
        wanted = []
        skipped = 0
        for entry in self.list_prefix():
            relative = self.relative_for(entry["key"])
            if not _matches_globs(relative, include, exclude):
                skipped += 1
                continue
            target = root / relative
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Download path escapes destination: {relative}")
            wanted.append((entry["key"], target, relative))

        def fetch(item):
            key, target, relative = item
            transferred = self.download_verified(key, target)
            print(f"{'downloaded' if transferred else 'present   '} {relative}", flush=True)
            return transferred

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
            transferred = sum(pool.map(fetch, wanted))
        return {"objects": len(wanted), "downloaded": transferred,
                "present": len(wanted) - transferred, "filtered_out": skipped}
