import geopandas
import numpy
from pathlib import Path
import rasterio

from river_profile_lidar.utils import (
    get_cropped_image_to_remove_black_padding,
    get_water_rgb_array_from_transect_df,
)
from river_profile_lidar.bathymetry.hab.utils import (
    calculate_da,
    rasterize_geometry,
    reproject_raster,
    get_basic_out_meta_data_tiff,
    smooth_transect_edges,
    save_array_as_raster,
)
from river_profile_lidar.water_speed_tools.utils import add_y0_to_transect_polygon_df, get_vji


def process_and_save_water_spped(
    rgb_image_path,
    transect_path,
    hab_image_path,
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
    water_mask = (water_rgb_array > 0).all(axis=2)
    water_rgb_array[~water_mask] = numpy.nan

    hab = rasterio.open(hab_image_path)
    hab_array = hab.read()
    hab_array = hab_array.transpose(1, 2, 0)

    # Step 3: Calculate transect hab distribution and speed
    transect_polygon_in_rgb["DA"] = transect_polygon_in_rgb.apply(
        lambda row: calculate_da(row["Q_IMG_spli"], row["Pente"], row["WAT_WIDTH"]),
        axis=1,
    )
    transect_polygon_in_rgb["Vi"] = (
        transect_polygon_in_rgb["Q_IMG_spli"] / transect_polygon_in_rgb["WAT_WIDTH"]
    )

    transect_polygon_in_rgb = add_y0_to_transect_polygon_df(
        transect_polygon_in_rgb,
        hab_array,
        hab.transform,
    )

    # Step 6: Rasterize and reproject y0 and slope
    raster_y0, transform_y0 = rasterize_geometry(
        transect_polygon_in_rgb,
        "y0",
        raster_resolution=raster_resolution,
    )

    raster_slope, transform_slope = rasterize_geometry(
        transect_polygon_in_rgb,
        "Pente",
        raster_resolution=raster_resolution,
    )

    raster_y0_reproject = reproject_raster(
        numpy.atleast_3d(raster_y0).transpose(2, 0, 1),
        transform_y0,
        profile["crs"],
        (1, height, width),
        affine_transform,
    )[0]

    raster_slope_reproject = reproject_raster(
        numpy.atleast_3d(raster_slope).transpose(2, 0, 1),
        transform_slope,
        profile["crs"],
        (1, height, width),
        affine_transform,
    )[0]

    # Step 7: Calculate water_speed estimate and smooth edges
    water_speed = get_vji(
        di=hab_array[:, :, 0], slope=raster_slope_reproject, y0=raster_y0_reproject
    )

    water_speed = smooth_transect_edges(
        transect_polygon_in_rgb,
        water_speed,
        raster_resolution,
        height,
        width,
        affine_transform,
        profile["crs"],
    )

    # Step 8: Save the final smoothed depth estimate raster
    water_speed = numpy.atleast_3d(water_speed).transpose(2, 0, 1)
    out_meta = get_basic_out_meta_data_tiff()
    save_array_as_raster(
        water_speed,
        out_meta,
        profile["crs"],
        affine_transform,
        Path(output_path, f"{image_name_prefix}_water_speed.tif"),
    )
