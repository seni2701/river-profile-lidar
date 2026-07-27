from warnings import warn
from affine import Affine
from rasterio.mask import mask as rasterio_mask
import numpy
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import Polygon
from shapely.ops import unary_union
import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile, DatasetReader
from rasterstats import zonal_stats
from skimage.util import img_as_float64
from typing import Tuple, Dict
from tqdm import tqdm


def find_bounding_box(image: numpy.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of non-zero pixels in an image.

    This function identifies the smallest rectangular bounding box that contains all
    non-zero pixels in a given image. The image is expected to have multiple bands,
    and the function considers pixels as non-zero if any of the first three bands
    have a non-zero value.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the image, with shape (bands, height, width).
        The function assumes the first three bands are used to determine non-zero pixels.

    Returns:
    --------
    xmin : int
        The minimum x-coordinate (column) of the bounding box.
    xmax : int
        The maximum x-coordinate (column) of the bounding box.
    ymin : int
        The minimum y-coordinate (row) of the bounding box.
    ymax : int
        The maximum y-coordinate (row) of the bounding box.
    """
    # Collapse the bands by checking for non-zero pixels across all bands
    non_zero_pixels = numpy.any(image[:3] != 0, axis=0)
    non_nan_pixels = ~numpy.any(numpy.isnan(image[:3]), axis=0)
    non_zero_pixels = non_zero_pixels * non_nan_pixels

    # Find rows and columns with any non-zero pixels
    non_zero_rows = numpy.any(non_zero_pixels, axis=1)
    non_zero_cols = numpy.any(non_zero_pixels, axis=0)

    # Identify the indices of the non-zero rows and columns
    ymin, ymax = numpy.where(non_zero_rows)[0][[0, -1]]
    xmin, xmax = numpy.where(non_zero_cols)[0][[0, -1]]
    return xmin, xmax, ymin, ymax


def get_cropped_image_to_remove_black_padding(
    rgb_image_path: str,
) -> Tuple[DatasetReader, Dict]:
    """
    Crop an image to remove black padding and return the cropped image with updated metadata.

    This function reads a multi-band image from the specified file path, identifies the bounding
    box of the non-black region, and crops the image to that bounding box. The function returns
    the cropped image along with its updated profile metadata.

    Parameters:
    -----------
    rgb_image_path : str
        The file path to the input image (e.g., a GeoTIFF file).

    Returns:
    --------
    cropped_image : rasterio.io.DatasetReader
        A dataset reader object representing the cropped image.
    profile : dict
        The updated profile (metadata) for the cropped image, including the new height,
        width, and affine transform.
    """
    with rasterio.open(rgb_image_path) as src:
        image = src.read()
        # transform = src.transform
        profile = src.profile
    xmin, xmax, ymin, ymax = find_bounding_box(image)
    cropped_image_array = image[:, ymin : ymax + 1, xmin : xmax + 1]
    window = Window.from_slices((ymin, ymax + 1), (xmin, xmax + 1))
    cropped_transform = src.window_transform(window)
    profile.update(
        {
            "height": cropped_image_array.shape[1],
            "width": cropped_image_array.shape[2],
            "transform": cropped_transform,
        }
    )
    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(cropped_image_array)
    cropped_image = memfile.open()
    return cropped_image, profile


def get_water_rgb_array_from_transect_df(
    rgb: DatasetReader, transect_polygon_df: GeoDataFrame
) -> Tuple[numpy.ndarray, Affine, GeoDataFrame, GeoDataFrame]:
    """
    Extract and crop RGB image data based on transects defined in a GeoDataFrame.

    This function reads an RGB image and a DataFrame containing transect polygons. It determines
    which transect polygons intersect with the bounds of the RGB image and then crops the image
    to the union of these intersecting transects. The function returns the cropped RGB image array,
    the affine transform for the cropped image, the subset of transect polygons within the image,
    and the updated DataFrame indicating which transects were within the RGB image bounds.

    Parameters:
    -----------
    rgb : rasterio.io.DatasetReader
        A rasterio dataset reader object representing the RGB image.
    transect_polygon_df : geopandas.geodataframe.GeoDataFrame
        A DataFrame containing transect polygons, with a "geometry" column holding shapely Polygon objects.

    Returns:
    --------
    water_rgb_array : numpy.ndarray
        A 3D numpy array of the cropped RGB image data, with shape (height, width, 3).
    affine_transform : affine.Affine
        The affine transform for the cropped RGB image.
    transect_polygon_in_rgb : geopandas.geodataframe.GeoDataFrame
        A DataFrame of transect polygons that intersect with the RGB image bounds.
    transect_polygon_df : geopandas.geodataframe.GeoDataFrame
        The original DataFrame with an additional column "is_in_rgb" indicating which transects
        are within the RGB image bounds.
    """
    rgb_bounds = rgb.bounds
    rgb_polygon = Polygon(
        [
            (rgb_bounds.left, rgb_bounds.top),
            (rgb_bounds.left, rgb_bounds.bottom),
            (rgb_bounds.right, rgb_bounds.bottom),
            (rgb_bounds.right, rgb_bounds.top),
            (rgb_bounds.left, rgb_bounds.top),
        ]
    )
    rgb_array = rgb.read()
    rgb_array = rgb_array.transpose(1, 2, 0)
    is_in_rgb_list = []
    for _, row in tqdm(
        transect_polygon_df.iterrows(),
        total=len(transect_polygon_df),
        desc="transect in RGB",
    ):
        specific_transect_polygon = row["geometry"]
        is_in_rgb = specific_transect_polygon.within(rgb_polygon)
        if is_in_rgb is True:
            transect_stats_list = []
            for band in range(0, 4):
                transect_stats = zonal_stats(
                    row["geometry"],
                    rgb_array[:, :, band],
                    affine=rgb.transform,
                    nodata=0,
                    stats=["min", "max"],
                )[0]
                transect_stats_list.append(transect_stats)
            all_band_min = numpy.array([stats["min"] for stats in transect_stats_list])
            all_band_max = numpy.array([stats["max"] for stats in transect_stats_list])
            if (all_band_min == 0).all() or (all_band_max == 255).all():
                is_in_rgb = False
        is_in_rgb_list.append(is_in_rgb)
    transect_polygon_df["is_in_rgb"] = is_in_rgb_list

    transect_polygon_in_rgb = transect_polygon_df[
        transect_polygon_df["is_in_rgb"]
    ].reset_index(drop=True)
    polygon_list = transect_polygon_in_rgb["geometry"].to_list()
    all_transect_polygon = unary_union(polygon_list)
    if len(polygon_list) == 0:
        warn(f"The image does not contain any transect polygons")
        return None, None, None, None
    water_rgb, affine_transform = rasterio_mask(rgb, [all_transect_polygon], crop=True)
    water_rgb_array = water_rgb[:3].transpose(1, 2, 0)
    water_rgb_array = img_as_float64(water_rgb_array)
    return (
        water_rgb_array,
        affine_transform,
        transect_polygon_in_rgb,
        transect_polygon_df,
    )


def get_touching_previous_transect_id_list(
    current_row, transect_polygon_df, pk_buffer_for_touching_polygon=50
):
    """
    From a row with a 'PK' and a 'geometry', return the ID list of previous (according to 'PK' value) touching transects
    """
    current_pk = current_row["PK"]
    current_geometry = current_row["geometry"]
    potential_previous_transect = transect_polygon_df[
        transect_polygon_df["PK"].between(
            current_pk - pk_buffer_for_touching_polygon, current_pk
        )
    ]
    touching_previous_row = potential_previous_transect.apply(
        lambda row: row["geometry"].touches(current_geometry)
        and row["PK"] < current_pk,
        axis=1,
    )
    id_list = list(potential_previous_transect[touching_previous_row]["ID"].values)
    return id_list


def get_touching_following_transect_id_list(
    current_row, transect_polygon_df, pk_buffer_for_touching_polygon=50
):
    """
    From a row with a 'PK' and a 'geometry', return the ID list of following (according to 'PK' value) touching transects
    """
    current_pk = current_row["PK"]
    current_geometry = current_row["geometry"]
    potential_following_transect = transect_polygon_df[
        transect_polygon_df["PK"].between(
            current_pk, current_pk + pk_buffer_for_touching_polygon
        )
    ]
    touching_previous_row = potential_following_transect.apply(
        lambda row: row["geometry"].touches(current_geometry)
        and row["PK"] > current_pk,
        axis=1,
    )
    id_list = list(potential_following_transect[touching_previous_row]["ID"].values)
    return id_list
