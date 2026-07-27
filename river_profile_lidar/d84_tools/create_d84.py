import geopandas
import numpy
from pathlib import Path

from river_profile_lidar.utils import (
    get_cropped_image_to_remove_black_padding,
    get_water_rgb_array_from_transect_df,
)
from river_profile_lidar.bathymetry.hab.utils import (
    rasterize_geometry,
    reproject_raster,
    get_basic_out_meta_data_tiff,
    save_array_as_raster,
)


def process_and_save_d84(
    rgb_image_path,
    transect_path,
    raster_resolution,
    output_path,
):
    # Step 1: Load and preprocess the image and transects
    image_name_prefix = Path(rgb_image_path).stem
    cropped_image, profile = get_cropped_image_to_remove_black_padding(rgb_image_path)
    transect_polygon_df = geopandas.read_file(transect_path)
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Backwater"] == 0]
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Lac"] == 0]
    transect_polygon_df = transect_polygon_df[~transect_polygon_df["Pente"].isna()]
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Pente"] > 0]
    transect_polygon_df = transect_polygon_df.sort_values(by="PK").reset_index(
        drop=True
    )

    # Step 2: Extract water RGB array and relevant data
    (
        water_rgb_array,
        affine_transform,
        transect_polygon_in_rgb,
        transect_polygon,
    ) = get_water_rgb_array_from_transect_df(cropped_image, transect_polygon_df)
    if transect_polygon_in_rgb is None:
        return None
    if len(transect_polygon_in_rgb) == 0:
        return None
    height, width, _ = water_rgb_array.shape

    # Step 6: Rasterize and reproject d84
    raster_d84, transform_d84 = rasterize_geometry(
        transect_polygon_in_rgb,
        "D84",
        raster_resolution=raster_resolution,
    )
    raster_d84_reproject = reproject_raster(
        numpy.atleast_3d(raster_d84).transpose(2, 0, 1),
        transform_d84,
        profile["crs"],
        (1, height, width),
        affine_transform,
    )[0]

    # Step 8: Save the final smoothed depth estimate raster
    raster_d84_reproject = numpy.atleast_3d(raster_d84_reproject).transpose(2, 0, 1)
    out_meta = get_basic_out_meta_data_tiff()
    save_array_as_raster(
        raster_d84_reproject,
        out_meta,
        profile["crs"],
        affine_transform,
        Path(output_path, f"{image_name_prefix}_d84.tif"),
    )
