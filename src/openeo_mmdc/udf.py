#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call MALICE model
for Sentinel-2 time series embeddings
"""
import sys
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube

# # Names for new 640 bands. In format Q..F..
# NEW_BANDS = [f"F0{i}" for i in range(10)] + [f"F{i}" for i in range(10, 64)]
NEW_BANDS = ["F01_mu", "F02_mu", "F03_mu", "F04_mu", "F05_mu", "F06_mu",
             "F01_logvar", "F02_logvar", "F03_logvar", "F04_logvar", "F05_logvar", "F06_logvar"]


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Apply metadata"""
    return metadata.rename_labels(
        dimension="band",
        target=NEW_BANDS
    )


def check_datacube(cube: xr.DataArray):
    """Check datacube """
    if cube.data.ndim != 4:
        raise RuntimeError("DataCube dimensions should be (t,bands, y, x)")

    if cube.data.shape[1] == 10 or cube.data.shape[1] == 6:
        raise RuntimeError(
            "DataCube should have at least 3 days of temporal series)"
        )


def s2_angles_processing(s2_angles_data: np.ndarray) -> np.ndarray:
    """
    Transform S2 angles from degrees to sin/cos
    """

    (sun_az, sun_zen, view_az, view_zen) = (
        s2_angles_data[:, 0, ...],
        s2_angles_data[:, 1, ...],
        s2_angles_data[:, 2, ...],
        s2_angles_data[:, 3, ...],
    )

    return np.stack(
        [
            np.cos(np.deg2rad(sun_zen)),
            np.cos(np.deg2rad(sun_az)),
            np.sin(np.deg2rad(sun_az)),
            np.cos(np.deg2rad(view_zen)),
            np.cos(np.deg2rad(view_az)),
            np.sin(np.deg2rad(view_az)),
        ],
        axis=1,
    )


def s1_angles_processing(angles_s1_asc: np.array, angles_s1_desc: np.array) -> np.array:
    """Transform S1 angles"""
    angles = np.concat((angles_s1_asc, angles_s1_desc), 1)

    processed_angles = np.zeros_like(angles)
    processed_angles[(angles != np.nan) | (angles != 0)] = np.cos(
        np.deg2rad(angles[(angles != np.nan) | (angles != 0)])
    )

    return processed_angles


def apply_log_to_s1(data: np.array) -> np.array:
    """
    Apply log to S1 data
    """
    clip_min: float = 1e-4
    clip_max: float = 2.0

    nan_idxs = ~np.isnan(data)
    data[nan_idxs] = np.log10(np.clip(data[nan_idxs], clip_min, clip_max))
    return data


def dem_height_aspect(dem_data: np.ndarray) -> np.ndarray:
    """
    compute the dem gradient, and then slope and aspect
    param : dem_data
    return : stack with height, sin slope, cos aspect, sin aspect
    """
    x, y = np.gradient(dem_data.astype(np.float32))  # pylint: disable=invalid-name
    slope = np.rad2deg(
        np.arctan(np.sqrt(x * x + y * y) / 10)
    )  # 10m = resolution of SRTM
    # Aspect unfolding rules from
    # pylint: disable=C0301
    # https://github.com/r-barnes/richdem/blob/603cd9d16164393e49ba8e37322fe82653ed5046/include/richdem/methods/terrain_attributes.hpp#L236  # noqa: E501
    aspect = np.rad2deg(np.arctan2(x, -y))
    lt_0 = aspect < 0
    gt_90 = aspect > 90
    remaining = np.logical_and(aspect >= 0, aspect <= 90)
    aspect[lt_0] = 90 - aspect[lt_0]
    aspect[gt_90] = 360 - aspect[gt_90] + 90
    aspect[remaining] = 90 - aspect[remaining]

    # calculate trigonometric representation
    sin_slope = np.sin(np.deg2rad(slope))
    cos_aspect = np.cos(np.deg2rad(aspect))
    sin_aspect = np.sin(np.deg2rad(aspect))

    return np.stack([dem_data, sin_slope, cos_aspect, sin_aspect])


def run_inference(input_data: np.ndarray, satellite: str = "s2") -> np.ndarray:
    """
    Inference function for Sentinel-2 embeddings with prosailVAE.
    The input should be in shape (B, C, H, W)
    The output shape is (B, 22, H, W)
    """
    # First get virtualenv
    sys.path.insert(0, "tmp/extra_venv")
    import onnxruntime as ort

    model_file = f"tmp/extra_files/mmdc_experts_{satellite}.onnx"

    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.use_deterministic_compute = True

    # Execute on cpu only
    ep_list = ["CPUExecutionProvider"]

    # Create inference session
    ort_session = ort.InferenceSession(model_file, sess_options=so, providers=ep_list)

    ro = ort.RunOptions()
    ro.add_run_config_entry("log_severity_level", "3")

    if satellite.lower() == "s2" or input_data.shape[1] == 63:
        # We transform input data in right format
        s2_ref, s2_angles, meteo, dem = (
            input_data[:, :10].astype(np.float32),
            input_data[:, 10:14].astype(np.float32),
            input_data[:, 14:14 + 48].astype(np.float32),
            input_data[:, -1].astype(np.float32)
        )

        image = s2_ref.clip(0, 15000)
        dem = np.stack([dem_height_aspect(d) for d in dem])
        angles = s2_angles_processing(s2_angles)

    elif input_data.shape[1] == 15:
        s1_asc, s1_desc, s1_angles_asc, s1_angles_desc, meteo, dem = (
            input_data[:, :2].astype(np.float32),
            input_data[:, 2:3].astype(np.float32),
            input_data[:, 3:5].astype(np.float32),
            input_data[:, 5:6].astype(np.float32),
            input_data[:, 6:6 + 48].astype(np.float32),
            input_data[:, -1].astype(np.float32)
        )
        image = np.concat(
            (
                s1_asc, s1_asc[:, 1] / s1_asc[:, 0] + 3e-05,
                s1_desc, s1_desc[:, 1] / s1_desc[:, 0] + 3e-05
            ),
            axis=1
        )
        image = apply_log_to_s1(image)
        dem = np.stack([dem_height_aspect(d) for d in dem])
        angles = s1_angles_processing(s1_angles_asc, s1_angles_desc)
    else:
        raise IndexError

    input = {
        "img": image,
        "angles": angles,
        "dem": dem,
        "meteo": meteo,
    }

    # Get the ouput of the exported model
    res_mu, res_logvar = ort_session.run(None, input, run_options=ro)
    result = np.concatenate((res_mu, res_logvar), axis=1, dtype=np.float32)
    return result


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
    encoded = run_inference(cubearray.data, context["satellite"])
    # Build output data array
    predicted_cube = xr.DataArray(
        encoded,
        dims=["t", "bands", "y", "x"],
        coords=cubearray.coords,
    )

    return XarrayDataCube(predicted_cube)
