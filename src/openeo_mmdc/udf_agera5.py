#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call MALICE model
for Sentinel-2 time series embeddings
"""
from typing import Dict

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube
NEW_BANDS = [f"F0{i}" for i in range(10)] + [f"F{i}" for i in range(10, 48)]

def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Apply metadata"""
    metadata = metadata.rename_labels(
        dimension="band",
        target=NEW_BANDS
    )
    return metadata


def check_datacube(cube: xr.DataArray):
    """Check datacube """
    if cube.data.ndim != 4:
        raise RuntimeError("DataCube dimensions should be (t,bands, y, x)")

    if cube.data.shape[1] == 10 or cube.data.shape[1] == 6:
        raise RuntimeError(
            "DataCube should have at least 3 days of temporal series)"
        )


def run_date_selection(
        input_data: np.ndarray, dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    meteo_data = input_data[:, :-1]
    mask_data = input_data[:, -1]
    where_ref_date = mask_data.sum((1, 2)) > 0
    ref_date = dates[where_ref_date]

    all_mini_series = [
        meteo_data[idx_date-4:idx_date+2].reshape(48, *mask_data.shape[-2:])
        for idx_date, valid in enumerate(where_ref_date) if valid
    ]

    # for idx_date, valid in enumerate(where_ref_date):
    #     if valid:
    #         mini_sits = meteo_data[idx_date-4:idx_date+2]
    #         mini_sits.reshape(48, *mini_sits.shape[2:])

    return np.ndarray(all_mini_series, dtype=np.float32), ref_date


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

    # Build output data array
    mini_series, ref_dates = run_date_selection(cubearray.data, cubearray.t.values)
    mini_series_cube = xr.DataArray(
        mini_series,
        dims=["t", "bands", "y", "x"],
        coords=cubearray.coords,
    )

    return XarrayDataCube(mini_series_cube)
