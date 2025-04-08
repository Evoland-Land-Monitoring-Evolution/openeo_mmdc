#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Extract dates for AGERA5 mini-time-series
"""
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Apply metadata"""
    # metadata = metadata.rename_labels(
    #     dimension="band",
    #     target=["mask_ref", "mask_filter"]
    # )

    return metadata


def compute_dates(cubearray: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    We compute dates for AGERA5 6 days mini-series.
    cubearray is masked data of shape T x 1 x H x W,
    where T is time dimension of the reference image data, T \in [t_1, ..., t_n]
    For each day t_n we compute the dates of AGERA5 mini-series that are [t_n-4: t_n+1]
    """

    # We extract the reference dates for time series
    ref_dates = cubearray.coords["t"].values

    # We get new dates in the following way:
    # For each day d_n, we extract days [d_n-4: d_n+1]
    new_t, indices = np.unique([t + np.timedelta64(dd, 'D')
                                for t in ref_dates for dd in range(-4, 2, 1)], return_index=True)

    # We mark the reference dates d_n as 1, other days as 0
    reference_index = np.array([t in ref_dates for t in new_t], dtype=np.float32)

    where_ref_date = np.array(np.broadcast_to(reference_index[:, None, None, None], (len(new_t), 1, *cubearray.shape[-2:])))
    # where_ref_date[:, 1, :, :] = 0
    return where_ref_date, new_t

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

    where_ref_date, new_t = compute_dates(cubearray)
    print(cubearray)
    # print(new_t)
    masked_cube = xr.DataArray(
        where_ref_date,
        dims=["t", "bands", "y", "x"],
        coords={"t": new_t, "y": cubearray.coords["y"], "x": cubearray.coords["x"]},
    )
    print(masked_cube)
    return XarrayDataCube(masked_cube)
