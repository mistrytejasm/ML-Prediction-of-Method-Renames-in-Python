r"""
load_master.py
Read-only access to the canonical master dataset for every later phase
(preprocessing, cross-validation, training, evaluation).

  verify_master()     checks the file's SHA-256 against config.MASTER_SHA256
                      and against the checksum stored in its schema
  load_master()       verifies, then reads the CSV (no writes, ever)
  select_xy_groups()  X for an approved feature set, the target and the
                      grouping column for grouped cross-validation

The master is never modified here: the file is only opened for reading.

Run as a check:  python src/data/load_master.py
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.features.feature_sets import get_feature_set, validate_feature_set  # noqa: E402

log = logging.getLogger("load_master")
GROUP_COLUMN = "meta_project_group"


class MasterChecksumError(RuntimeError):
    """The master dataset on disk is not the audited, canonical file."""


def sha256_of(path, chunk=1 << 20):
    """SHA-256 of a file, read in chunks (read-only)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_master(path=config.MASTER, schema_path=config.MASTER_SCHEMA, expected=config.MASTER_SHA256):
    """Return the checksum if `path` is the canonical master; raise MasterChecksumError otherwise."""
    path = Path(path)
    if not path.exists():
        raise MasterChecksumError(f"master dataset not found: {path}")
    actual = sha256_of(path)
    if actual != expected:
        raise MasterChecksumError(f"{path} has SHA-256 {actual}, expected {expected}. "
                                  "The canonical master dataset must not be modified.")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    stored = schema.get("sources", {}).get("output_sha256")
    if stored != expected:
        raise MasterChecksumError(f"schema {schema_path} records SHA-256 {stored}, expected {expected}")
    return actual


def load_schema(schema_path=config.MASTER_SCHEMA):
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def load_master(path=config.MASTER, verify=True):
    """Read the master dataset after verifying its checksum. Empty `class` stays '' (module-level functions)."""
    if verify:
        verify_master(path)
    df = pd.read_csv(path, keep_default_na=False)
    df["class"] = df["class"].astype(str)
    return df


def select_xy_groups(df, feature_set="main"):
    """Return (X, y, groups) for an approved feature set: 'main' (40) or 'comparison' (42).

    X holds only approved predictive features; identifiers, the target and every
    meta_* column (including the grouping column) are kept out of X.
    """
    features = validate_feature_set(get_feature_set(feature_set), allow_file_size=(feature_set != "main"))
    missing = [c for c in features + [config.TARGET, GROUP_COLUMN] if c not in df.columns]
    if missing:
        raise KeyError(f"columns missing from the dataset: {missing}")
    return df[features], df[config.TARGET], df[GROUP_COLUMN]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    digest = verify_master()
    df = load_master(verify=False)
    log.info("Master verified: SHA-256 %s", digest)
    for name in ("main", "comparison"):
        X, y, groups = select_xy_groups(df, name)
        log.info("%-10s X %s | positives %d | groups %d", name, X.shape, int(y.sum()), groups.nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
