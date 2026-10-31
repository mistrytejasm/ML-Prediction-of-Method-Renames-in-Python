"""
Stage 8: write-once, auditable run folders under results/runs/.

  run_directory_name(prefix, run_id, day)  "<YYYY-MM-DD>_<prefix>_<run_id>"
  write_run(...)                           successful run -> final folder
  write_failed_run(...)                    failed run -> "<name>__FAILED" folder (no OOF,
                                           status "failed"), so a failure is recorded but
                                           can never be mistaken for a result
  provenance(...)                          git commit + dirty flag, versions, hashes
  stale_temporary_runs(runs_dir)           leftovers of interrupted writes

Every folder is first written to "<runs_dir>/.tmp_<name>_<random>" and renamed to its
final name only after every file has been written. An existing folder is never
overwritten (the final rename refuses), and the temporary folder is removed if
anything fails. Folders whose name starts with ".tmp_" are never results.

Successful run folder:
  run_config.json      the RunSpec (model, resolved params, seeds, feature set, transform, status)
  provenance.json      git commit and dirty flag, Python / package versions, master SHA-256,
                       split hash, split CSV SHA-256, fold statistics, start / end time, status
  fold_metrics.csv     one row per fold
  oof_predictions.csv  master_row, commit, file, class, name, meta_project_group, fold,
                       renamed, oof_probability (one row per development row, master order)
  summary.json         CV AP (mean, std ddof=1, per fold), threshold selection, optimistic
                       development threshold metrics, fold threshold diagnostics
  warnings.json        every captured warning with its fold
No fitted model is stored.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.load_master import sha256_of  # noqa: E402

REQUIRED_FILES = ("run_config.json", "provenance.json", "fold_metrics.csv", "oof_predictions.csv",
                  "summary.json", "warnings.json")
FAILED_SUFFIX = "__FAILED"
TEMP_PREFIX = ".tmp_"
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm")


class RunStoreError(RuntimeError):
    """A run folder could not be written without overwriting or losing data."""


def run_directory_name(prefix, run_id, day=None):
    day = day or date.today().isoformat()
    name = f"{day}_{prefix}_{run_id}"
    if any(ch in name for ch in '/\\:*?"<>|') or name.startswith("."):
        raise RunStoreError(f"invalid run directory name {name!r}")
    return name


def _json(obj):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.as_posix()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")
    return json.dumps(obj, indent=2, default=default, allow_nan=True) + "\n"


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=True, timeout=30)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state():
    """Commit and whether tracked or untracked files differ from it (results/runs/ is ignored for this)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".", ":(exclude)results/runs")
    return {"commit": commit, "dirty": None if status is None else bool(status),
            "dirty_paths": None if not status else status.splitlines()[:50]}


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def provenance(split_info, extra=None):
    lock = config.ROOT / "requirements-lock.txt"
    return {
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "requirements_lock_sha256": sha256_of(lock) if lock.exists() else None,
        "master_dataset": config.MASTER.relative_to(config.ROOT).as_posix(),
        "master_sha256": split_info.get("dataset_sha256"),
        "split_hash": split_info.get("split_hash"),
        "split_csv_sha256": split_info.get("assignment_file_sha256"),
        "development_counts": split_info.get("development"),
        "fold_statistics": split_info.get("folds"),
        "command": sys.argv,
        **(extra or {}),
    }


def write_folder(runs_dir, name, files):
    """Write `files` (name -> writer(path)) to runs_dir/name via a temporary folder; never overwrites."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    final = runs_dir / name
    if final.exists():
        raise RunStoreError(f"{final} already exists; runs are never overwritten")
    tmp = runs_dir / f"{TEMP_PREFIX}{name}_{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        for fname, writer in files.items():
            writer(tmp / fname)
        try:
            os.rename(tmp, final)                          # refuses an existing target on every platform
        except OSError as exc:
            raise RunStoreError(f"could not move the run to {final}: {exc}") from exc
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def text_file(content):
    return lambda path: path.write_text(content, encoding="utf-8", newline="\n")


def write_run(runs_dir, name, spec, result, prov):
    """Write a successful run; returns the final folder."""
    started = prov.get("started_utc")
    prov = {**prov, "status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
            "started_utc": started}
    files = {
        "run_config.json": text_file(_json(asdict(spec))),
        "provenance.json": text_file(_json(prov)),
        "fold_metrics.csv": lambda p: result["fold_metrics"].to_csv(p, index=False, lineterminator="\n"),
        "oof_predictions.csv": lambda p: result["oof"].to_csv(p, index=False, lineterminator="\n"),
        "summary.json": text_file(_json(result["summary"])),
        "warnings.json": text_file(_json(result["warnings"])),
    }
    if tuple(files) != REQUIRED_FILES:
        raise RunStoreError("run files differ from REQUIRED_FILES")
    return write_folder(runs_dir, name, files)


def write_failed_run(runs_dir, name, spec, error, warnings_log, prov):
    """Record a failed run as '<name>__FAILED' with its error and warnings, but no results."""
    prov = {**prov, "status": "failed", "finished_utc": datetime.now(timezone.utc).isoformat()}
    files = {"run_config.json": text_file(_json(asdict(spec))), "provenance.json": text_file(_json(prov)),
             "error.json": text_file(_json({"error": str(error), "type": type(error).__name__,
                                        "fold": getattr(error, "fold", None)})),
             "warnings.json": text_file(_json(warnings_log))}
    return write_folder(runs_dir, name + FAILED_SUFFIX, files)


def is_complete_run(folder):
    """True only for a finished, successful run folder."""
    folder = Path(folder)
    if folder.name.startswith(TEMP_PREFIX) or folder.name.endswith(FAILED_SUFFIX) or not folder.is_dir():
        return False
    if not all((folder / f).exists() for f in REQUIRED_FILES):
        return False
    return json.loads((folder / "provenance.json").read_text(encoding="utf-8")).get("status") == "complete"


def stale_temporary_runs(runs_dir):
    runs_dir = Path(runs_dir)
    return sorted(p for p in runs_dir.glob(f"{TEMP_PREFIX}*") if p.is_dir()) if runs_dir.exists() else []
