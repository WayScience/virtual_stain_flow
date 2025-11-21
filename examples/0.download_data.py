"""
Download JUMP pilot plate data from AWS S3 bucket.

This script downloads high-resolution cell imaging data from the JUMP (JCP
Understanding Morphology & Plasticity) public dataset. The data includes
multiple channels of brightfield and fluorescence microscopy images.

Data Source:
    S3 bucket: s3://cellpainting-gallery/cpg0000-jump-pilot/
    Access: Public (no AWS credentials required with --no-sign-request)
    Reference: https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1

Defaults:
    Batch: 2020_11_04_CPJUMP1
    Plate: BR00117010__2020-11-08T18_18_00-Measurement1
    Output: ./jump_pilot_subset/

Usage:
    # Download with defaults
    python 0.download_data.py
    
    # Download to custom directory
    python 0.download_data.py --outdir /path/to/data
    
    # Download different plate
    python 0.download_data.py --batch 2020_11_04_CPJUMP1 --plate PLATE_ID

Requirements:
    - AWS CLI installed: https://aws.amazon.com/cli/
    - Internet connection
    - Sufficient disk space (~10GB for full plate)

Note: Initial download may take several minutes depending on internet speed.
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
    parser = argparse.ArgumentParser(
        description="Download JUMP pilot plate data from AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 0.download_data.py
  python 0.download_data.py --outdir ~/data/jump
  python 0.download_data.py --batch 2020_11_04_CPJUMP1 --plate BR00117010__2020-11-08T18_18_00-Measurement1
        """
    )
    parser.add_argument(
        "--batch",
        type=str,
        default='2020_11_04_CPJUMP1',
        help="JUMP batch ID (default: 2020_11_04_CPJUMP1)"
    )
    parser.add_argument(
        "--plate",
        type=str,
        default='BR00117010__2020-11-08T18_18_00-Measurement1',
        help="Plate ID within batch (default: BR00117010__2020-11-08T18_18_00-Measurement1)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="jump_pilot_subset",
        help="Output directory for downloaded data (default: ./jump_pilot_subset)"
    )
    args = parser.parse_args()
    download_jump_plate(args.batch, args.plate, outdir=args.outdir)


if __name__ == "__main__":
    main()
