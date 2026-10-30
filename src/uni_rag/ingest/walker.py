"""Metadata-only discovery. Never follows symlinks or opens document contents."""

import fnmatch
import os
import stat
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from uni_rag.config import Config
from uni_rag.models import FileRecord

SF_DATALESS = 0x40000000
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".obsidian",
    ".trash",
    "archives",
    "archive",
}
PERMISSION_HINT = (
    "Allow your terminal app in System Settings → Privacy & Security → Files and Folders "
    "(or Full Disk Access)."
)


@dataclass
class Scan:
    files: list[FileRecord] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    seen: int = 0

    def skip(self, source: str, path: str, reason: str, detail: str = "") -> None:
        self.skipped.append({"source": source, "path": path, "reason": reason, "detail": detail})

    @property
    def skip_counts(self) -> dict[str, int]:
        return dict(Counter(row["reason"] for row in self.skipped))

    def summary(self) -> dict:
        groups: dict[str, dict] = {}
        folders: Counter = Counter()
        for item in self.files:
            key = f"{item.source}:{item.path.suffix.lower()}"
            group = groups.setdefault(key, {"files": 0, "bytes": 0})
            group["files"] += 1
            group["bytes"] += item.size
            folders[f"{item.source}:{PurePosixPath(item.relative_path).parent}"] += item.size
        return {
            "files_seen": self.seen,
            "eligible_files": len(self.files),
            "by_type": groups,
            "largest_folders_bytes": dict(folders.most_common(20)),
            "skipped_by_reason": self.skip_counts,
        }


def walk(
    config: Config, source: str | None = None, only: str | None = None, limit: int | None = None
) -> Scan:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    prefix = PurePosixPath(only or ".")
    if prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError("--only must be a relative subfolder without '..'")
    result = Scan()
    for src in config.selected_sources(source):
        root = src.path
        if root.is_symlink():
            result.skip(src.name, ".", "symlink")
            continue
        start = root / prefix
        # Reject symlinks in each --only component before traversing them.
        cursor = root
        for part in prefix.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("--only must not traverse a symlink")
        pending = [start]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    ordered = sorted(entries, key=lambda entry: entry.name)
            except (PermissionError, OSError) as exc:
                reason = "permission" if isinstance(exc, PermissionError) else "unavailable"
                result.skip(
                    src.name,
                    str(directory.relative_to(root)),
                    reason,
                    f"{exc}. {PERMISSION_HINT}" if reason == "permission" else str(exc),
                )
                continue
            for entry in ordered:
                rel = str(Path(entry.path).relative_to(root))
                try:
                    if entry.is_symlink():
                        result.skip(src.name, rel, "symlink")
                        continue
                    info = entry.stat(follow_symlinks=False)
                    is_dir = stat.S_ISDIR(info.st_mode)
                    if not is_dir:
                        result.seen += 1
                    if any(
                        fnmatch.fnmatch(rel, p)
                        or fnmatch.fnmatch(entry.name, p)
                        or PurePosixPath(rel).match(p)
                        for p in config.exclude_patterns
                    ):
                        result.skip(src.name, rel, "excluded_pattern")
                    elif is_dir and entry.name.lower() in EXCLUDED_DIRS:
                        result.skip(src.name, rel, "excluded_directory")
                    elif entry.name.endswith(".icloud"):
                        result.skip(src.name, rel, "icloud_placeholder")
                    elif config.skip_cloud_only and getattr(info, "st_flags", 0) & SF_DATALESS:
                        result.skip(src.name, rel, "cloud_only")
                    elif is_dir:
                        pending.append(Path(entry.path))
                    elif not stat.S_ISREG(info.st_mode):
                        result.skip(src.name, rel, "not_regular")
                    elif Path(entry.name).suffix.lower() not in config.include_extensions:
                        result.skip(src.name, rel, "extension")
                    elif info.st_size > config.max_file_mb * 1024 * 1024:
                        result.skip(src.name, rel, "too_large")
                    elif limit is not None and len(result.files) >= limit:
                        result.skip(src.name, rel, "limit")
                    else:
                        result.files.append(
                            FileRecord(
                                src.name, Path(entry.path), rel, info.st_size, info.st_mtime_ns
                            )
                        )
                except OSError as exc:
                    result.skip(
                        src.name,
                        rel,
                        "permission" if isinstance(exc, PermissionError) else "stat_error",
                        f"{exc}. {PERMISSION_HINT}",
                    )
    return result
