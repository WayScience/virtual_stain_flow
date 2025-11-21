"""
Example script to download a JUMP pilot plate from S3 bucket.
Defaults to batch '2020_11_04_CPJUMP1' and plate 'BR00117010__2020-11-08T18_18_00-Measurement1'.

Usage:
    python download_jump.py
"""

import subprocess
from pathlib import Path

BASE = "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images"

def download_jump_plate(batch, plate, outdir="jump_pilot_subset"):
    out = Path(outdir) / batch / plate
    out.mkdir(parents=True, exist_ok=True)

    s3_prefix = f"{BASE}/{batch}/images/{plate}/Images/"
    print("Downloading from:", s3_prefix)

    subprocess.run(
        [
            "aws", "s3", "cp", "--recursive",
            s3_prefix, str(out),
            "--no-sign-request",
        ],
        check=True,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default='2020_11_04_CPJUMP1', required=True, help="Batch ID, e.g. batch1")
    parser.add_argument("--plate", type=str, default='BR00117010__2020-11-08T18_18_00-Measurement1', required=True, help="Plate ID, e.g. plate1")
    parser.add_argument("--outdir", type=str, default="jump_pilot_subset")
    args = parser.parse_args()
    download_jump_plate(args.batch, args.plate, outdir=args.outdir)


if __name__ == "__main__":
    main()
