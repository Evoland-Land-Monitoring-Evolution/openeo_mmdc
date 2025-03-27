#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Extract dates for AGERA5 mini-time-series
"""
from typing import Dict

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Apply metadata"""
    metadata = metadata.rename_labels(
        dimension="band",
        target=["mask"]
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
    ref_dates = cubearray.coords["t"].values
    reference_index = np.array([[0, 0, 0, 0, 1, 0]
                                for _ in ref_dates], dtype=np.float32).reshape(-1)
    new_t, indices = np.unique([t + np.timedelta64(dd, 'D')
                                for t in ref_dates for dd in range(-4, 2, 1)], return_index=True)
    reference_index = reference_index[indices]

    where_ref_date = np.broadcast_to(reference_index[:, None, None, None], (len(new_t), 1, *cubearray.shape[-2:]))
    print(new_t)
    masked_cube = xr.DataArray(
        where_ref_date,
        dims=["t", "bands", "y", "x"],
        coords={"t": new_t, "y": cubearray.coords["y"], "x": cubearray.coords["x"]},
    )

    return XarrayDataCube(masked_cube)
