import geopandas
import numpy
from pathlib import Path
import rasterio
from rasterio.features import geometry_mask

from river_profile_lidar.utils import (
    get_cropped_image_to_remove_black_padding,
    get_water_rgb_array_from_transect_df,
)
from river_profile_lidar.bathymetry.hab.utils import (
    add_pixel_distribution_to_transect_polygon_df,
    calculate_da,
    add_beta_and_max_pixel_value,
    rasterize_geometry,
    smooth_beta_and_max_value_with_touching_transect,
    reproject_raster,
    caculate_depth_value,
    detect_and_interpolate_bad_water_pixels,
    get_basic_out_meta_data_tiff,
    smooth_transect_edges,
    save_array_as_raster,
    interpolate_bathymetry_in_shaded_areas,
    get_points_in_shade,
    replace_bathymetry_in_too_much_shade,
)
from river_profile_lidar.bathymetry.trapezoid.utils import (
    fix_nan_pixels_with_gaussian_convolution,
)


def process_and_save_hab(
    rgb_image_path,
    transect_path,
    apply_sobel,
    shade_kwargs,
    raster_resolution,
    output_path,
):
    """
    Process bathymetry data by removing black padding from the image, extracting transect polygons,
    interpolating bad pixels, calculating depth estimates, and saving the results as raster files.

    Parameters:
    -----------
    rgb_image_path : str
        Path to the RGB image file.
    transect_path : str
        Path to the transect shapefile.
    apply_sobel : bool
        Whether to apply Sobel filtering and interpolation to remove bad pixels.
    shade_kwargs : dict
        kwargs for shade. Needs the
    raster_resolution : float
        The resolution of the raster output.
    output_path : str
        Path to save the final smoothed depth estimate raster file.
    """
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
    if shade_kwargs is not None:
        handle_shade = True
        assert "shade_mask_path" in shade_kwargs.keys()
        assert "trapezoid_path" in shade_kwargs.keys()
        assert "cross_section_points_path" in shade_kwargs.keys()
        assert "min_proportion_of_not_in_shade_pixels" in shade_kwargs.keys()
    else:
        handle_shade = False

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
    if handle_shade:
        tree_shade_df = geopandas.read_file(shade_kwargs["shade_mask_path"])
        trapezoid = rasterio.open(shade_kwargs["trapezoid_path"])
        shade_mask = geometry_mask(
            tree_shade_df.geometry.values,
            out_shape=(height, width),
            transform=affine_transform,
            invert=True,
        )
        trapezoid_array = reproject_raster(
            trapezoid.read(),
            trapezoid.transform,
            trapezoid.crs,
            water_rgb_array.shape[:2],
            affine_transform,
        )
        water_rgb_array = water_rgb_array * (~numpy.atleast_3d(shade_mask))
        trapezoid_in_shade_array = trapezoid_array.copy()
        trapezoid_in_shade_array[~shade_mask] = numpy.nan
    else:
        trapezoid_in_shade_array = None

    water_mask = (water_rgb_array > 0).all(axis=2)
    water_rgb_array[~water_mask] = numpy.nan

    # Step 3: Detect and interpolate bad pixels if required
    if apply_sobel:
        (
            water_rgb_array,
            is_water_bad_pixel_mask,
        ) = detect_and_interpolate_bad_water_pixels(
            water_rgb_array,
            water_mask,
            sobel_threshold=0.03,
            sobel_gaussian_std=1,
            sobel_gaussian_threshold=0.25,
        )

        water_rgb_array_bad_pixel = water_rgb_array.copy().astype(float)
        water_rgb_array_bad_pixel[~is_water_bad_pixel_mask] = None
        water_rgb_array_bad_pixel_mask = water_rgb_array_bad_pixel[:, :, 0]
        water_rgb_array_bad_pixel_mask[~numpy.isnan(water_rgb_array_bad_pixel_mask)] = 1
        water_rgb_array_bad_pixel_mask = numpy.atleast_3d(
            water_rgb_array_bad_pixel_mask
        ).transpose(2, 0, 1)
        out_meta = get_basic_out_meta_data_tiff()
        save_array_as_raster(
            water_rgb_array_bad_pixel_mask,
            out_meta,
            profile["crs"],
            affine_transform,
            Path(output_path, f"{image_name_prefix}_bad_pixel_mask.tif"),
        )

        water_rgb_array_out = water_rgb_array.transpose(2, 0, 1)
        save_array_as_raster(
            water_rgb_array_out,
            out_meta,
            profile["crs"],
            affine_transform,
            Path(output_path, f"{image_name_prefix}_interpolated_pixels.tif"),
        )

    # Step 5: Calculate transect pixel distribution and depth
    transect_polygon_in_rgb = add_pixel_distribution_to_transect_polygon_df(
        transect_polygon_in_rgb,
        water_rgb_array,
        affine_transform,
        trapezoid_in_shade_array,
    )

    transect_polygon_in_rgb["DA"] = transect_polygon_in_rgb.apply(
        lambda row: calculate_da(row["Q_IMG_spli"], row["Pente"], row["WAT_WIDTH"]),
        axis=1,
    )
    transect_polygon_in_rgb = add_beta_and_max_pixel_value(
        transect_polygon_in_rgb, adjust_for_shade=handle_shade
    )
    transect_polygon_in_rgb = smooth_beta_and_max_value_with_touching_transect(
        transect_polygon_in_rgb, 7
    )

    # Step 6: Rasterize and reproject beta and max pixel values
    raster_beta_smoothed, transform_beta_smoothed = rasterize_geometry(
        transect_polygon_in_rgb, "beta_smoothed", raster_resolution=raster_resolution
    )
    raster_max_value_smoothed, transform_max_value_smoothed = rasterize_geometry(
        transect_polygon_in_rgb,
        "max_pixel_value_smoothed",
        raster_resolution=raster_resolution,
    )

    raster_beta_reproject_smoothed = reproject_raster(
        numpy.atleast_3d(raster_beta_smoothed).transpose(2, 0, 1),
        transform_beta_smoothed,
        profile["crs"],
        (1, height, width),
        affine_transform,
    )[0]

    raster_max_value_reproject_smoothed = reproject_raster(
        numpy.atleast_3d(raster_max_value_smoothed).transpose(2, 0, 1),
        transform_max_value_smoothed,
        profile["crs"],
        (1, height, width),
        affine_transform,
    )[0]

    # Step 7: Calculate depth estimate and smooth edges
    depth_estimate = caculate_depth_value(
        water_rgb_array[:, :, 0].astype(float),
        raster_max_value_reproject_smoothed,
        raster_beta_reproject_smoothed,
    )
    depth_estimate[~water_mask] = None
    depth_estimate[depth_estimate < 0] = 0

    # Step 8: Shade Handling
    # Start by remplacing the transect that do not fit the min_proportion_of_not_in_shade_pixels
    # by the trapezoid depth
    all_transect_in_too_much_shade_mask = replace_bathymetry_in_too_much_shade(
        transect_polygon_in_rgb,
        height,
        width,
        affine_transform,
        shade_kwargs["min_proportion_of_not_in_shade_pixels"],
    )
    depth_estimate[all_transect_in_too_much_shade_mask] = trapezoid_array[
        all_transect_in_too_much_shade_mask
    ]
    remaining_shade_mask = shade_mask * ~all_transect_in_too_much_shade_mask

    # Interpolated values in shade
    if handle_shade and remaining_shade_mask.sum() > 0:
        points_in_image_and_shade_df = get_points_in_shade(
            tree_shade_df,
            shade_kwargs["cross_section_points_path"],
            transect_polygon_in_rgb,
        )
        (
            depth_estimate_shade,
            affine_transform_shade,
        ) = interpolate_bathymetry_in_shaded_areas(
            transect_polygon_in_rgb,
            depth_estimate,
            remaining_shade_mask,
            height,
            width,
            affine_transform,
            points_in_image_and_shade_df,
            raster_resolution,
        )
        if depth_estimate_shade is not None:
            depth_estimate_shade = reproject_raster(
                numpy.atleast_3d(depth_estimate_shade).transpose(2, 0, 1),
                affine_transform_shade,
                profile["crs"],
                (1, height, width),
                affine_transform,
            )[0]
            depth_estimate[remaining_shade_mask] = depth_estimate_shade[
                remaining_shade_mask
            ]

    water_mask_for_interpolation_projection = geometry_mask(
        transect_polygon_in_rgb.geometry.values,
        out_shape=(depth_estimate.shape[0], depth_estimate.shape[1]),
        transform=affine_transform,
        invert=True,
    )
    depth_estimate = fix_nan_pixels_with_gaussian_convolution(
        depth_estimate, water_mask_for_interpolation_projection
    )

    depth_estimate = smooth_transect_edges(
        transect_polygon_in_rgb,
        depth_estimate,
        raster_resolution,
        height,
        width,
        affine_transform,
        profile["crs"],
    )

    # Step 9: Save the final smoothed depth estimate raster
    depth_estimate = numpy.atleast_3d(depth_estimate).transpose(2, 0, 1)
    out_meta = get_basic_out_meta_data_tiff()
    save_array_as_raster(
        depth_estimate,
        out_meta,
        profile["crs"],
        affine_transform,
        Path(output_path, f"{image_name_prefix}_hab.tif"),
    )
