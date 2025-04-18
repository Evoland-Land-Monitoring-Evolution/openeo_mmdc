# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
MMDC embeddings with OpenEO
"""

import argparse
import datetime
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from openeo import DataCube

DEPENDENCIES_URL: str = (
    "https://artifactory.vgt.vito.be:443/auxdata-public/openeo/onnx_dependencies.zip"
)

import openeo
from openeo_mmdc import __version__
from shapely import geometry

__author__ = "Ekaterina Kalinicheva"
__copyright__ = "Ekaterina Kalinicheva"
__license__ = "AGPL-3.0-or-later"

_logger = logging.getLogger(__name__)


def default_bands_list_agera5() -> list[str]:
    return [
        "dewpoint-temperature",
        "precipitation-flux",
        "solar-radiation-flux",
        "temperature-max",
        "temperature-mean",
        "temperature-min",
        "vapour-pressure",
        "wind-speed",
    ]


def default_bands_list(satellite: str) -> list[str]:
    """Default bands for each satellite"""
    if satellite.lower() == "s2":
        return ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    return ["VV", "VH"]


def default_angles_list(satellite: str) -> list[str]:
    if satellite.lower() == "s2":
        return ["sunAzimuthAngles", "sunZenithAngles", "viewAzimuthMean", "viewZenithMean"]
    return ["local_incidence_angle"]  # For S1 incidence angle is defined in .sar_backscatter(...)


@dataclass(frozen=True)
class Parameters:
    """Collection parameters"""
    spatial_extent: Dict[str, float]
    start_date: str
    end_date: str
    output_file: str
    max_cloud_cover: int = 30
    openeo_instance: str = "openeo.vito.be"
    collection: str = "SENTINEL2_L2A_SENTINELHUB"
    satellite: str = "s2"
    patch_size: int = 256
    overlap: int = 28


def process(parameters: Parameters, output: str) -> None:
    """
    Main processing function
    """
    # First connect to OpenEO instance
    print("Connection...")
    connection = openeo.connect(parameters.openeo_instance).authenticate_oidc()

    # Search for the S2 datacube
    if parameters.satellite.lower() == "s2":
        MODEL_URL: str = \
            "https://artifactory.vgt.vito.be/artifactory/evoland/mmdc_models/mmdc_experts_s2.zip"
        sat_cube: DataCube = connection.load_collection(
            parameters.collection,
            spatial_extent=parameters.spatial_extent,
            temporal_extent=[parameters.start_date, parameters.end_date],
            bands=default_bands_list(parameters.satellite) + default_angles_list(parameters.satellite),
            max_cloud_cover=parameters.max_cloud_cover,
            fetch_metadata=True
        )

        print("Create mask")
        mask_dates = sat_cube.band("B02") * 0

    else:
        MODEL_URL: str = \
            "https://artifactory.vgt.vito.be/artifactory/evoland/mmdc_models/mmdc_experts_s1.zip"
        sat_cube = None

        for orbit in ["ASCENDING", "DESCENDING"]:
            sat_cube_orbit: DataCube = connection.load_collection(
                parameters.collection,
                spatial_extent=parameters.spatial_extent,
                temporal_extent=[parameters.start_date, parameters.end_date],
                bands=default_bands_list(parameters.satellite),
                fetch_metadata=True,
                properties={"sat:orbit_state": lambda od: od == orbit}
            ).sar_backscatter(
                coefficient="gamma0-terrain",
                elevation_model=None,
                mask=False,
                contributing_area=False,
                local_incidence_angle=True,
                ellipsoid_incidence_angle=False,
                noise_removal=True,
                options=None,
            )

            sat_cube_orbit = sat_cube_orbit.rename_labels(dimension="bands", target=["VV_" + orbit, "VH_" + orbit,
                                                                                     "local_incidence_angle_" + orbit])

            if sat_cube is None:
                sat_cube = sat_cube_orbit
            else:
                sat_cube = sat_cube.merge_cubes(sat_cube_orbit)
            print("metadata s1", sat_cube_orbit.metadata)

        udf_file_match_s1 = os.path.join(os.path.dirname(__file__), f"udf_find_match_s1.py")
        udf_match_s1 = openeo.UDF.from_file(udf_file_match_s1, runtime="Python-Jep")
        sat_cube = sat_cube.apply_dimension(udf_match_s1)

        print("Create mask")
        mask_dates = sat_cube.band("VV_ASCENDING") * 0

    # We create a mask with reference dates for AGERA5 6 days mini-series
    # 1 - day d
    # 0 - days d-4, d-3, d-2, d-1, d+1
    mask_dates = mask_dates.add_dimension(name="bands", label="mask", type="bands")
    print("mask dates", mask_dates.metadata)

    udf_file_t = os.path.join(os.path.dirname(__file__), f"udf_t.py")
    udf_time = openeo.UDF.from_file(udf_file_t, runtime="Python-Jep")
    mask_for_agera5 = mask_dates.apply_neighborhood(udf_time, size=[
        {"dimension": "x",
         "value": 256,
         "unit": "px"},
        {"dimension": "y",
         "value": 256,
         "unit": "px"},
    ], overlap=[])


    print(f"Get AGERA... "
          f"For each available day d of {parameters.satellite.upper()} image, "
          f"get 6 days mini-series [d-4:d+1] of weather data with 8 variables")
    start_meteo = (
            datetime.datetime.strptime(parameters.start_date, '%Y-%m-%d') - datetime.timedelta(days=4)
    ).strftime('%Y-%m-%d')
    end_meteo = (
            datetime.datetime.strptime(parameters.end_date, '%Y-%m-%d') + datetime.timedelta(days=1)
    ).strftime('%Y-%m-%d')

    agera5 = connection.load_collection(
        "AGERA5",
        temporal_extent=[start_meteo, end_meteo],
        bands=default_bands_list_agera5(),
    )
    print("metadata", agera5.metadata)

    # Get the same spatio-temporal extent as S1/S2 images
    geometry_box = geometry.box(
        parameters.spatial_extent["west"],
        parameters.spatial_extent["south"],
        parameters.spatial_extent["east"],
        parameters.spatial_extent["north"])

    agera5 = agera5.resample_cube_spatial(
        sat_cube, method="cubic"
    ).filter_spatial(geometry_box)

    # Filter datacube dates (does not work)
    agera5 = agera5.mask(mask_for_agera5 * 0)

    agera5 = agera5.merge_cubes(mask_for_agera5)
    print("metadata", agera5.metadata)

    # Handle optional overlap parameter
    overlap = []
    if parameters.overlap is not None:
        overlap = [
            {"dimension": "x", "value": parameters.overlap, "unit": "px"},
            {"dimension": "y", "value": parameters.overlap, "unit": "px"},
        ]

    # 6 days mini-series of AGERA5 with 8 variables
    # Reshape them to T x 48 x H x W, where T is length of image SITS
    udf_file_agera5 = os.path.join(os.path.dirname(__file__), f"udf_agera5.py")
    udf_agera5 = openeo.UDF.from_file(udf_file_agera5, runtime="Python-Jep")
    mini_agera5 = agera5.apply_neighborhood(udf_agera5, size=[
        {"dimension": "x",
         "value": 256,
         "unit": "px"},
        {"dimension": "y",
         "value": 256,
         "unit": "px"},
    ], overlap=[])

    # Get DEM and do spatio-temporal resampling as S1/S2
    dem = connection.load_collection(
        "COPERNICUS_30",
        bands=["DEM"],
    )
    dem = dem.resample_cube_spatial(sat_cube, method="cubic").filter_spatial(geometry_box)

    # Merge all datacubes S1/S2 + AGERA5 + DEM
    sat_cube = sat_cube.merge_cubes(mini_agera5).merge_cubes(dem.max_time())
    print("metadata", sat_cube.metadata)

    # job = sat_cube.create_job(
    #     title=f"mmdc_{parameters.satellite}_collection",
    #     description=f"mmdc_{parameters.satellite}",
    #     out_format="netCDF",
    #     sample_by_feature=False,
    #     job_options=job_options,
    # )
    # job.start_job()

    # Process the final cube with the inference UDF
    udf_file = os.path.join(os.path.dirname(__file__), f"udf.py")
    udf = openeo.UDF.from_file(udf_file, runtime="Python-Jep", context={"from_parameter": "context"})
    job_options = {
        "udf-dependency-archives": [
            f"{DEPENDENCIES_URL}#tmp/extra_venv",
            f"{MODEL_URL}#tmp/extra_files",
        ],
        "executor-memory": "10G",
        "executor-memoryOverhead": "20G",  # default 2G
        "executor-cores": 2,
        "task-cpus": 1,
        "executor-request-cores": "400m",
        "max-executors": "100",
        "driver-memory": "16G",
        "driver-memoryOverhead": "16G",
        "driver-cores": 5,
        "logging-threshold": "info",
    }

    mmdc_sat_cube = sat_cube.apply_neighborhood(
        udf,
        size=[
            {"dimension": "x",
             "value": parameters.patch_size - parameters.overlap * 2,
             "unit": "px"},
            {"dimension": "y",
             "value": parameters.patch_size - parameters.overlap * 2,
             "unit": "px"},
        ],
        overlap=overlap,
        context={"satellite": parameters.satellite.lower()}
    )

    # Download embeddings
    download_job1 = mmdc_sat_cube.save_result("netCDF").create_job(
        title=f"mmdc_{parameters.satellite}", job_options=job_options
    )
    download_job1.start_and_wait()
    os.makedirs(output, exist_ok=True)
    download_job1.get_results().download_files(output)

    # Download the original images
    download_job2 = mmdc_sat_cube.save_result("netCDF").create_job(title="sits-orig")
    download_job2.start_and_wait()
    os.makedirs(os.path.join(output, "original"), exist_ok=True)
    download_job2.get_results().download_files(output)


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Run Single-Image prosailVAE embedding model on OpenEO and download results"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openeo_mmdc {__version__}",
    )
    parser.add_argument(
        "--satellite", help="Modality we want to encode. Should be s1 or s2",
        type=str, required=True
    )
    parser.add_argument(
        "--start_date", help="Start date (format: YYYY-MM-DD)", type=str, required=True
    )
    parser.add_argument(
        "--end_date", help="End date (format: YYYY-MM-DD)", type=str, required=True
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        help="Extent (west lat, east lat, south lon, north lon)",
        required=True,
    )
    parser.add_argument(
        "--output", type=str, help="Path to ouptput NetCDF file", required=True
    )

    parser.add_argument(
        "--instance",
        type=str,
        default="https://openeo.vito.be/openeo/1.2",
        help="OpenEO instance on which to run the MALICE algorithm",
    )
    return parser.parse_args(args)


def main(args):
    """
    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)

    # Build parameters
    parameters = Parameters(
        spatial_extent={
            "west": args.extent[0],
            "east": args.extent[1],
            "south": args.extent[2],
            "north": args.extent[3],
        },
        openeo_instance=args.instance,
        start_date=args.start_date,
        end_date=args.end_date,
        collection="SENTINEL2_L2A_SENTINELHUB"
        if args.satellite.lower() == "s2" else "SENTINEL1_GRD",
        satellite=args.satellite.lower(),
        output_file=args.output,
    )
    print(parameters)

    _logger.info(f"Parameters : {parameters}")
    process(parameters, args.output)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m openeo_mmdc.skeleton 42
    #
    run()
