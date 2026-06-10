#!/usr/bin/env python
# coding: utf-8

# # Download JUMP pilot plate data from AWS S3 bucket for example training

# In[1]:


from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split

from virtual_stain_flow.datasets.example.cpjump1_manifest import get_manifest
from virtual_stain_flow.datasets.example.arrange_as_wide import arrange_manifest_channels


# ## Pathing

# In[ ]:


DATA_DOWNLOAD_DIR = Path("/PATH/TO/WHERE/YOU/WANT/TO/DOWNLOAD/CPJUMP1")
DATA_DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)


# ## S3 download helpers

# In[3]:


def _parse_s3_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URL, got: {url}")
    return parsed.netloc, parsed.path.lstrip("/")

def download_wide_manifest_channels(
    wide_manifest,
    dest_dir,
    channel_columns=None,
    overwrite=False,
):
    """
    Download S3 TIFFs for each channel and write a local file_index.csv with paths.
    """
    if channel_columns is None:
        channel_columns = ["LZ_BF", "BF", "HZ_BF", "DNA", "Mito", "AGP", "ER", "RNA"]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 downloads. Install with: pip install boto3"
        ) from exc
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    local_rows = []
    for row_idx, row in wide_manifest.iterrows():
        prefix_parts = []
        for key in ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]:
            if key in wide_manifest.columns:
                prefix_parts.append(str(row[key]))
        prefix = "_".join(prefix_parts) if prefix_parts else f"row_{row_idx}"
        local_row = {}
        for channel in channel_columns:
            url = row[channel] if channel in wide_manifest.columns else None
            if pd.isna(url):
                local_row[channel] = None
                continue
            bucket, key = _parse_s3_url(url)
            suffix = Path(key).suffix or ".tif"
            local_path = dest_dir / f"{prefix}_{channel}{suffix}"
            if overwrite or not local_path.exists():
                s3.download_file(bucket, key, str(local_path))
            local_row[channel] = str(local_path)
        local_rows.append(local_row)
    file_index = pd.DataFrame(local_rows, columns=channel_columns)
    file_index.to_csv(dest_dir / "file_index.csv", index=False)
    return file_index


# ## Retrieve compound manifest

# In[4]:


MANIFEST = get_manifest()
MANIFEST.head()


# ## Filter manifest
# For the sake of demoing training here we restricted timepoint to 24, and selected untreated U2-OS cells

# In[5]:


negcon_a549_48_manifest = MANIFEST[
    (MANIFEST["Cell_type"] == "A549") &
    (MANIFEST["Anomaly"] == "none") &
    (MANIFEST["control_type"] == 'negcon') &
    (MANIFEST["Time"] == 48)
]
negcon_a549_48_manifest.head()


# ## Arrange as wide is the anticipated format in virtual stain flow datasets and also the format the download helper expects this format

# In[6]:


wide_manifest = arrange_manifest_channels(negcon_a549_48_manifest)
wide_manifest.head()


# ## Data split

# In[7]:


a549_data_dir = DATA_DOWNLOAD_DIR / "cpjump1_a549_48h"
a549_data_dir.mkdir(exist_ok=True, parents=True)

# Get unique plates
unique_plates = wide_manifest['Metadata_Plate'].unique()

# Split plates into train (75%) and test (25%) with seed
train_plates, test_plates = train_test_split(
    unique_plates, 
    test_size=0.25, 
    random_state=42
)

# Create train and test manifests based on plate split
train_manifest_wide = wide_manifest[wide_manifest['Metadata_Plate'].isin(train_plates)]
test_manifest_wide = wide_manifest[wide_manifest['Metadata_Plate'].isin(test_plates)]

print(f"Train plates: {len(train_plates)}, Test plates: {len(test_plates)}")
print(f"Train samples: {len(train_manifest_wide)}, Test samples: {len(test_manifest_wide)}")

negcon_a549_48_manifest.to_csv(a549_data_dir / "raw_manifest.csv", index=False)
train_manifest_wide.to_csv(a549_data_dir/ "train_manifest.csv", index=False)
test_manifest_wide.to_csv(a549_data_dir / "test_manifest.csv", index=False)


# ## Download all data

# In[ ]:


_ = download_wide_manifest_channels(
    train_manifest_wide,
    dest_dir = a549_data_dir / "train"    
)
_ = download_wide_manifest_channels(
    test_manifest_wide,
    dest_dir = a549_data_dir / "test"    
)

