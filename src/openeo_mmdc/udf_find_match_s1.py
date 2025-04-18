#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Match S1 ASC and DESC images by date
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Apply metadata"""
    return metadata.rename_labels(
        dimension="band",
        target=['VV_ASCENDING', 'VH_ASCENDING', 'local_incidence_angle_ASCENDING',
                'VV_DESCENDING', 'VH_DESCENDING', 'local_incidence_angle_DESCENDING']
    )


def match_asc_desc_both_available(
        days_asc: np.ndarray, days_desc: np.ndarray, tolerance: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match Asc et Desc dates when both orbits are available.
    Tolerance is +-1 day
    """
    df_asc = pd.DatetimeIndex(days_asc).to_frame(name="asc")
    df_desc = pd.DatetimeIndex(days_desc).to_frame(name="desc")
    s1_asc_desc = pd.merge_asof(
        df_asc,
        df_desc,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(tolerance, "D"),
        direction="nearest",
    )
    s1_desc_asc = pd.merge_asof(
        df_desc,
        df_asc,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(tolerance, "D"),
        direction="nearest",
    )

    df_merged_days = (
        pd.concat([s1_asc_desc, s1_desc_asc]).sort_index().drop_duplicates()
    )
    days_s1 = df_merged_days.index.values
    print(days_s1)
    df_merged_days = df_merged_days.reset_index()

    # Get indices of days of each orbit within all the available S1 dates
    asc_ind = df_merged_days.index[df_merged_days["asc"].notnull()].values
    desc_ind = df_merged_days.index[df_merged_days["desc"].notnull()].values
    return asc_ind, desc_ind, days_s1


def match_asc_desc(
        days_asc: np.ndarray, days_desc: np.ndarray, tolerance: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match asc and desc orbits to find co-ocurences
    Returns Asc and Desc dates indices within all available S1 dates
    and days_s1 - all available dates
    """
    # if both orbits are available for patch
    if len(days_asc) > 0 and len(days_desc) > 0:
        return match_asc_desc_both_available(days_asc, days_desc, tolerance)
    # if only one orbit is available for patch
    else:
        if len(days_asc) > 0:
            asc_ind = np.arange(len(days_asc))
            days_s1 = days_asc
            desc_ind = []
        else:
            desc_ind = np.arange(len(days_desc))
            days_s1 = days_desc
            asc_ind = []

    return asc_ind, desc_ind, days_s1


def run_matching(input_data: xr.DataArray, tolerance: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match S1 ASC and S1 DESC dates
    """
    dates = input_data.coords["t"].values

    dates = np.array([np.datetime64(str(d)) for d in dates])

    valid_s1_asc = (np.isnan(input_data.data[:, 0])).mean((1, 2)) < 0.98
    valid_s1_desc = (np.isnan(input_data.data[:, 3])).mean((1, 2)) < 0.98

    valid_dates_s1_asc = dates[valid_s1_asc]
    valid_dates_s1_desc = dates[valid_s1_desc]

    asc_ind, desc_ind, days_s1 = match_asc_desc(valid_dates_s1_asc, valid_dates_s1_desc, tolerance=tolerance)

    new_data = np.full((len(days_s1), 6, *input_data.data.shape[-2:]), np.nan)

    new_data[asc_ind, :3] = input_data.data[valid_s1_asc, :3]
    new_data[desc_ind, 3:] = input_data.data[valid_s1_desc, 3:]

    return new_data.astype(np.float32), days_s1


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply UDF function to datacube
    """
    # We get data from datacube
    cubearray: xr.DataArray
    if isinstance(cube, xr.DataArray):
        cubearray = cube
    else:
        cubearray = cube.get_array().copy()
    matched, new_t = run_matching(cubearray)

    # Build output data array
    predicted_cube = xr.DataArray(
        matched,
        dims=cubearray.dims,
        coords={"t": new_t, "y": cubearray.coords["y"], "x": cubearray.coords["x"]},
    )
    return XarrayDataCube(predicted_cube)
