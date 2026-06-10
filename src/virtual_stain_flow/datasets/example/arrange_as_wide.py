"""
Helper utility specifically to support the example CPJUMP1 dataset 
    pivoting and arranging as file index.
"""

import pandas as pd


def arrange_manifest_channels(manifest):
    """
    Return a wide dataframe with one row per plate/well/site and URL columns per channel.
    """
    required_channels = ["LZ_BF", "BF", "HZ_BF", "DNA", "Mito", "AGP", "ER", "RNA"]
    keys = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
    filtered = manifest[manifest["Metadata_ChannelName"].isin(required_channels)].copy()
    filtered["Metadata_ChannelName"] = filtered["Metadata_ChannelName"].astype(
        pd.CategoricalDtype(categories=required_channels, ordered=True)
    )
    filtered = filtered.sort_values(keys + ["Metadata_ChannelName"])
    wide = (
        filtered.pivot_table(
            index=keys,
            columns="Metadata_ChannelName",
            values="Metadata_FileUrl",
            aggfunc="first",
            observed=False,
        )
        .reindex(columns=required_channels)
        .reset_index()
    )
    
    return wide
