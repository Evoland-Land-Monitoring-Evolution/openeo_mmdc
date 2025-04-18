#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call MMDC model
for Sentinel-1/2 image embeddings
"""
import sys
from typing import Dict

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube

# Names for embedding band names
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

    if cube.data.shape[1] == 55 or cube.data.shape[1] == 63:
        raise RuntimeError(
            "Wrong number of input bands"
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
    angles = np.concatenate((angles_s1_asc, angles_s1_desc), 1)

    processed_angles = angles.copy()
    processed_angles[(processed_angles != np.nan) | (processed_angles != 0)] = np.cos(
        np.deg2rad(angles[(processed_angles != np.nan) | (processed_angles != 0)])
    )

    return processed_angles


def apply_log_to_s1(data: np.array) -> np.array:
    """
    Apply log to S1 data, ignore nans
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
    Inference function for Sentinel-1/2 embeddings with MMDC.
    The input should be in shape (B, C, H, W), where C_S1=55 and C_S2=63
    The output shape is (B, 12, H, W)
    The embeddings come in shape of distribution, we have 6 distributions in total.
    Each is characterized by its mean and logvar
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

    # We transform input data in right format
    if satellite.lower() == "s2" or input_data.shape[1] == 63:
        image, s2_angles, meteo, dem = (
            input_data[:, :10],
            input_data[:, 10:14],
            input_data[:, 14:14 + 48],
            input_data[:, -1]
        )

        dem = np.stack([dem_height_aspect(d) for d in dem])
        angles = s2_angles_processing(s2_angles)

    elif satellite.lower() == "s1" or input_data.shape[1] == 55:
        input_data[input_data == np.inf] = np.nan
        input_data[input_data == -np.inf] = np.nan

        s1_asc, s1_angles_asc, s1_desc, s1_angles_desc, meteo, dem = (
            input_data[:, :2],
            input_data[:, 2:3],
            input_data[:, 3:5],
            input_data[:, 5:6],
            input_data[:, 6:6 + 48],
            input_data[:, -1]
        )
        s1_asc[s1_asc == 0] = np.nan
        s1_desc[s1_desc == 0] = np.nan

        s1_angles_asc[s1_angles_asc == 0] = np.nan
        s1_angles_desc[s1_angles_desc == 0] = np.nan


        # We compute VH/VV ratio for both ASC and DESC orbits
        nan_asc = ~(np.isnan(s1_asc).sum(1) > 0)
        nan_desc = ~(np.isnan(s1_desc).sum(1) > 0)

        ratio_asc = np.full_like(nan_asc, np.nan, dtype=np.float32)
        ratio_desc = np.full_like(nan_desc, np.nan, dtype=np.float32)

        ratio_asc[nan_asc] = (s1_asc[:, 1] / s1_asc[:, 0] + 1e-05)[nan_asc]
        ratio_desc[nan_desc] = (s1_desc[:, 1] / s1_desc[:, 0] + 1e-05)[nan_desc]

        image = np.concatenate(
            [
                s1_asc, ratio_asc[:, None],
                s1_desc, ratio_desc[:, None]
            ],
            axis=1
        )

        image = apply_log_to_s1(image)
        dem = np.stack([dem_height_aspect(d) for d in dem])
        angles = s1_angles_processing(s1_angles_asc, s1_angles_desc)

    else:
        raise IndexError

    input = {
        "img": image.astype(np.float32),
        "angles": angles.astype(np.float32),
        "dem": dem.astype(np.float32),
        "meteo": meteo.astype(np.float32),
    }

    # Get the ouput of the exported model
    print("starting inference")
    res_mu, res_logvar = ort_session.run(None, input, run_options=ro)
    return np.concatenate((res_mu, res_logvar), axis=1, dtype=np.float32)


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

    satellite = context["satellite"]
    encoded = run_inference(cubearray.data, satellite)

    # Build output data array
    predicted_cube = xr.DataArray(
        encoded,
        dims=["t", "bands", "y", "x"],
        coords={"t": cubearray.coords["t"], "y": cubearray.coords["y"], "x": cubearray.coords["x"]},
    )

    return XarrayDataCube(predicted_cube)
