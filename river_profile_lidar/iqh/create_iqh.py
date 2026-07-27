import numpy
from pathlib import Path
import rasterio
import pickle


from river_profile_lidar.bathymetry.hab.utils import (
    get_basic_out_meta_data_tiff,
    save_array_as_raster,
)


def process_and_save_iqh(
    image_prefix_name,
    hab_image_path,
    water_speed_image_path,
    d84_image_path,
    output_path,
):
    # Step 1: Load and preprocess the images
    hab = rasterio.open(hab_image_path)
    hab_array = hab.read(1)
    water_speed = rasterio.open(water_speed_image_path)
    water_speed_array = water_speed.read(1)
    d84 = rasterio.open(d84_image_path)
    d84_array = d84.read(1)

    with open("habitat_model.pkl", "rb") as f:
        habitat_model = pickle.load(f)

    # Step 2: Calculate IQH
    water_mask = ~numpy.isnan(hab_array)
    iqh_1d = habitat_model.get_hist_1d_estimation_from_value(d84_array)
    iqh_2d = habitat_model.get_hist_2d_estimation_from_value(
        hab_array, water_speed_array
    )
    iqh = iqh_1d * iqh_2d
    iqh[~water_mask] = numpy.nan

    # Step 3: Save the final raster
    iqh = numpy.atleast_3d(iqh).transpose(2, 0, 1)
    out_meta = get_basic_out_meta_data_tiff()
    output_path.mkdir(parents=True, exist_ok=True)

    save_array_as_raster(
        iqh,
        out_meta,
        hab.crs,
        hab.transform,
        Path(output_path, f"{image_prefix_name}_iqh.tif"),
    )
