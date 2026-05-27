"""Build an enriched image manifest for CPJUMP1 dataset access.

Only compound perturbations (no CRISPR or ORF) are included, which is
appropriate for virtual staining experiments.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd

# Most recent commit ref as of Mar 25 2026.
REPO_REF = "6ea3958c3809cd04ac95b63138937dd64a7c4c12"
REPO_BASE = f"https://github.com/WayScience/JUMP-single-cell/raw/{REPO_REF}/"

IMAGE_MANIFEST_URL = f"{REPO_BASE}0.download_data/data/2020_11_04_CPJUMP1_all_plates.parquet"
IMAGE_MANIFEST_COLUMNS = [
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_ChannelName",
    "Metadata_PlaneID",
    "Metadata_PositionZ",
    "Metadata_FileUrl",
    "Metadata_Filename",
]

EXPERIMENT_METADATA_URL = f"{REPO_BASE}reference_plate_data/experiment-metadata.tsv"
COMPOUND_PLATEMAP_URL = f"{REPO_BASE}reference_plate_data/JUMP-Target-1_compound_platemap.txt"
COMPOUND_METADATA_URL = f"{REPO_BASE}reference_plate_data/JUMP-Target-1_compound_metadata_targets.tsv"

__all__ = ["build_manifest", "get_manifest", "main"]

_MANIFEST_CACHE: Optional[pd.DataFrame] = None


def build_manifest() -> pd.DataFrame:
    """
    Main utility function that handles all the wrangling. 
    Return an enriched CPJUMP1 manifest as a pandas DataFrame.
    """
    image_manifest = pd.read_parquet(IMAGE_MANIFEST_URL, columns=IMAGE_MANIFEST_COLUMNS)

    experiment_meta = pd.read_csv(EXPERIMENT_METADATA_URL, delimiter="\t")
    experiment_meta.rename(columns={"Assay_Plate_Barcode": "Metadata_Plate"}, inplace=True)
    experiment_meta = experiment_meta[experiment_meta["Perturbation"] == "compound"]
    # exclude dl batch which is essentially duplicate in context for image data access
    experiment_meta = experiment_meta[~experiment_meta["Batch"].str.endswith("_DL")]

    compound_platemap = pd.merge(
        pd.read_csv(COMPOUND_PLATEMAP_URL, delimiter="\t"),
        pd.read_csv(COMPOUND_METADATA_URL, delimiter="\t"),
        on="broad_sample",
        how="left",
        validate="many_to_one"    
    ).rename(columns={"well_position": "Metadata_Well"}, inplace=False)

    image_manifest_compound = pd.merge(
        experiment_meta,
        image_manifest,
        on="Metadata_Plate",
        how="inner",
        validate="one_to_many" # one plate id should map to many image rows
    )

    return pd.merge(
        compound_platemap,
        image_manifest_compound,
        on="Metadata_Well",
        how="inner",
        # all the plates share the same well map so one well should map to many image rows
        validate="one_to_many" 
    )


def get_manifest() -> pd.DataFrame:
    """
    Return a cached manifest to avoid repeated network reads.
    """
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = build_manifest()
    return _MANIFEST_CACHE


def _write_manifest(df: pd.DataFrame, output: str, fmt: str) -> None:
    if fmt == "csv":
        df.to_csv(output, index=False)
    elif fmt == "parquet":
        df.to_parquet(output, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Command-line interface to building and ouputting the CPJUMP1 manifest.
    By default, it prints a summary and preview of the manifest. 
    Use --output or --stdout to write the full manifest to a file or stdout.
    May or may not be useful. 
    """
    parser = argparse.ArgumentParser(description="Build CPJUMP1 enriched manifest.")
    parser.add_argument(
        "--output",
        help="Write manifest to a file (CSV or Parquet).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format when using --output (default: csv).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write manifest to stdout as CSV.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Rows to display when no output is specified (default: 5).",
    )
    args = parser.parse_args(argv)

    manifest = get_manifest()

    if args.stdout:
        manifest.to_csv(sys.stdout, index=False)
        return 0

    if args.output:
        _write_manifest(manifest, args.output, args.format)
        return 0

    print(f"Rows: {len(manifest):,} | Columns: {len(manifest.columns)}")
    print(manifest.head(args.head).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
