"""
Listing the raw ActRef commit folders.

A commit folder should be named with its 40-character commit hash. A folder
whose name is a hash plus a suffix (e.g. 'dd7f9fa8..._<garbled text>') and
whose hash also exists as a normal folder is a duplicate copy. It is skipped
only when its content is byte-identical to the original; otherwise it is kept
and reported, because it may be real data.
"""
import hashlib
import re
from pathlib import Path

HASH = re.compile(r"^[0-9a-f]{40}$")


def _digest(folder):
    h = hashlib.sha256()
    for f in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != ".DS_Store"):
        h.update(f.relative_to(folder).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def list_commit_dirs(commits_dir):
    """Return (kept folders sorted by name, {skipped folder name: reason})."""
    dirs = sorted(d for d in Path(commits_dir).iterdir() if d.is_dir())
    names = {d.name for d in dirs}
    kept, skipped = [], {}
    for d in dirs:
        prefix = d.name[:40]
        if not HASH.match(d.name) and HASH.match(prefix) and prefix in names:
            if _digest(d) == _digest(Path(commits_dir) / prefix):
                skipped[d.name] = f"byte-identical copy of commit folder {prefix}"
                continue
        kept.append(d)
    return kept, skipped
