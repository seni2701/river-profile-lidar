from affine import Affine
import numpy
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from rasterio.features import geometry_mask
from shapely.geometry import Point, Polygon
from geopandas.geodataframe import GeoDataFrame
from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import Gaussian2DKernel, convolve
from typing import List, Tuple
from tqdm import tqdm
import warnings


def create_z_interpolation_grid_from_points_list_and_polygon(
    points_list: List[Point], polygon: Polygon, raster_resolution: float
) -> Tuple[numpy.ndarray, Affine]:
    """
    Create a Z-value interpolation grid from a list of 3D points and a polygon boundary.

    This function generates a 2D grid representing the Z-values (depth)
    over a specified area defined by a polygon. The grid is created by interpolating
    the Z-values from a given list of 3D points using a linear method. The resulting
    grid is masked so that only areas within the polygon have valid data, and
    negative Z-values are set to zero.

    Parameters:
    -----------
    points_list : list of shapely.geometry.Point
        A list of shapely Point objects, each with x, y, and z coordinates
        representing the 3D points used for interpolation.
    polygon : shapely.geometry.Polygon
        A shapely Polygon object that defines the boundary within which the
        interpolation is performed.
    raster_resolution : float
        The spatial resolution of the output raster grid. Defines the distance
        between grid points in both x and y directions.

    Returns:
    --------
    z_mesh : numpy.ndarray
        A 2D array representing the interpolated Z-values within the polygon.
        Values outside the polygon are set to NaN.
    transform : affine.Affine
        The affine transformation matrix that maps pixel coordinates to geographic
        coordinates for the resulting grid.
    """
    points_x = [geo.x for geo in points_list]
    points_y = [geo.y for geo in points_list]
    points_z = [geo.z for geo in points_list]

    minx, miny, maxx, maxy = polygon.bounds
    x_grid = numpy.arange(minx, maxx, raster_resolution)
    y_grid = numpy.arange(miny, maxy, raster_resolution)
    x_mesh, y_mesh = numpy.meshgrid(x_grid, y_grid)
    boundary_coords = numpy.column_stack((points_x, points_y))
    z_mesh = griddata(boundary_coords, points_z, (x_mesh, y_mesh), method="linear")[
        ::-1, :
    ]
    transform = from_origin(minx, maxy, raster_resolution, raster_resolution)
    height, width = z_mesh.shape
    mask_polygon = geometry_mask(
        [polygon], out_shape=(height, width), transform=transform
    )
    z_mesh[mask_polygon] = numpy.nan
    z_mesh[z_mesh < 0] = 0
    return z_mesh, transform


def combine_list_of_array_and_transform_in_one_raster(
    array_list: List[numpy.ndarray], transform_list: Affine
) -> Tuple[numpy.ndarray, Affine]:
    """
    Combine multiple raster arrays and their corresponding transforms into a single merged raster.

    This function takes a list of 2D arrays (raster data) and their associated affine transformation
    matrices, and combines them into a single raster array. The resulting raster will cover the
    combined spatial extent of all input rasters, with overlapping areas merged by overlaying
    the arrays in the order provided.

    Parameters:
    -----------
    array_list : list of numpy.ndarray
        A list of 2D arrays representing raster data. Each array should correspond to a different
        section of the spatial area, with potentially overlapping regions.
    transform_list : list of affine.Affine
        A list of affine transformation matrices, one for each array in `array_list`. These transforms
        map the pixel coordinates of the arrays to geographic coordinates.

    Returns:
    --------
    merged_array : numpy.ndarray
        A 2D array representing the merged raster data. The size of this array will be large enough
        to cover the combined extents of all input arrays.
    merged_transform : affine.Affine
        The affine transformation matrix for the merged raster. This transform maps the pixel
        coordinates of `merged_array` to geographic coordinates.
    """
    # Determine the bounds of the combined raster
    min_x = min(transform.c for transform in transform_list)
    max_x = max(
        transform.c + array.shape[1] * transform.a
        for array, transform in zip(array_list, transform_list)
    )
    min_y = min(
        transform.f + array.shape[0] * transform.e
        for array, transform in zip(array_list, transform_list)
    )
    max_y = max(transform.f for transform in transform_list)

    # Create an empty array for the merged raster
    pixel_size = transform_list[0].a  # Assuming uniform pixel size for all rasters
    merged_width = round((max_x - min_x) / pixel_size)
    merged_height = round((max_y - min_y) / pixel_size)

    merged_array = numpy.full(
        (merged_height, merged_width), numpy.nan, dtype=array_list[0].dtype
    )
    # Define the transform for the merged raster
    merged_transform = from_origin(min_x, max_y, pixel_size, pixel_size)

    # Overlay each array onto the merged array
    for array, transform in zip(array_list, transform_list):
        x_offset = int((transform.c - min_x) / pixel_size)
        y_offset = int((max_y - transform.f) / pixel_size)
        mask = ~numpy.isnan(array)
        merged_array[
            y_offset : y_offset + array.shape[0], x_offset : x_offset + array.shape[1]
        ][mask] = array[mask]
    return merged_array, merged_transform


def process_transects_to_create_z_grids(
    transect_polygon_df: GeoDataFrame,
    points: GeoDataFrame,
    raster_resolution: float,
    potentiel_pk_difference: int = 50,
    min_distance: float = 0.01,
    min_points_to_use: int = 5,
):
    """
    Process transects from a GeoDataFrame to create Z-value interpolation grids for each transect polygon.

    This function iterates over a GeoDataFrame containing transect polygons and corresponding point data.
    For each transect polygon, it finds nearby points within a specified range, filters out duplicates,
    and generates a Z-value interpolation grid based on these points. The function returns lists of Z-value
    arrays and affine transformations for each processed transect polygon.

    Parameters:
    -----------
    transect_polygon_df : geopandas.geodataframe.GeoDataFrame
        A GeoDataFrame containing transect polygons with at least two columns:
        - "PK": A unique identifier for each transect polygon.
        - "geometry": A shapely Polygon object representing the transect polygon.
    points : geopandas.geodataframe.GeoDataFrame
        A DataFrame containing point data with at least two columns:
        - "PK": A unique identifier matching those in `transect_polygon_df`.
        - "geometry": A shapely Point object representing the point's location.
    raster_resolution : float
        The resolution of the raster grid used for Z-value interpolation.
    potentiel_pk_difference : int, optional
        The range around the current PK to consider for finding nearby points (default is 50).
    min_distance : float, optional
        The maximum distance for a point to be considered within the transect polygon (default is 0.01).
    min_points_to_use : int, optional
        The minimum number of points required to create a valid Z-value grid (default is 5).

    Returns:
    --------
    z_array_list : list of numpy.ndarray
        A list of 2D arrays representing the Z-value interpolation grids for each transect polygon.
    transform_list : list of affine.Affine
        A list of affine transformation matrices corresponding to each Z-value interpolation grid.
    """
    z_array_list = []
    transform_list = []

    for _, row in tqdm(
        transect_polygon_df.iterrows(),
        total=len(transect_polygon_df),
        desc="create depth grid",
    ):
        current_pk = row["PK"]
        current_polygon = row.geometry
        points_in_polygon = []

        potential_points_transect = points[
            points["PK"].isin(
                numpy.arange(
                    current_pk - potentiel_pk_difference,
                    current_pk + potentiel_pk_difference,
                    5,
                )
            )
        ]
        for points_index, points_row in potential_points_transect.iterrows():
            if points_row.geometry.distance(current_polygon) < min_distance:
                points_in_polygon.append(points_index)

        points_to_use = points.loc[points_in_polygon]
        points_to_use = points_to_use.drop_duplicates("geometry")

        if len(points_to_use) < min_points_to_use:
            continue
        try:
            (
                z_mesh,
                transform,
            ) = create_z_interpolation_grid_from_points_list_and_polygon(
                points_to_use.geometry.to_list(),
                current_polygon,
                raster_resolution=raster_resolution,
            )
        except Exception:
            continue

        z_array_list.append(z_mesh)
        transform_list.append(transform)

    return z_array_list, transform_list


def fix_nan_pixels_with_gaussian_convolution(
    merged_array, water_mask, gaussian_std=1, gaussian_size=21
):
    """
    Fix NaN pixels in an array using Gaussian convolution.

    This function iteratively applies Gaussian convolution to an input array to fill NaN values
    in areas defined by a water mask. The convolution continues until all NaN values within the
    water mask are replaced.

    Parameters:
    -----------
    merged_array : numpy.ndarray
        A 2D numpy array containing data with some NaN values.
    water_mask : numpy.ndarray
        A boolean mask array of the same shape as `merged_array`, where True values indicate
        regions to be processed (e.g., water regions).
    gaussian_std : float, optional
        The standard deviation of the Gaussian kernel in both x and y directions (default is 1).
    gaussian_size : int, optional
        The size of the Gaussian kernel (default is 21). It determines the width and height
        of the kernel.

    Returns:
    --------
    merged_array_fixed : numpy.ndarray
        A 2D numpy array with NaN values replaced by convolved values, based on the water mask.
    """
    # Define Gaussian kernel for convolution
    gaussian_kernel = Gaussian2DKernel(
        x_stddev=gaussian_std,
        y_stddev=gaussian_std,
        x_size=gaussian_size,
        y_size=gaussian_size,
    )

    # Make a copy of the array to avoid modifying the original
    merged_array_no_nan = merged_array.copy()
    nan_pixel_sum = numpy.isnan(merged_array_no_nan[water_mask]).sum()
    # Apply Gaussian convolution until all NaN values within the water mask are filled
    with tqdm(
        total=nan_pixel_sum, desc="Fix Remaining NaN Pixels with Gaussian Convolution"
    ) as pbar:
        while nan_pixel_sum > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AstropyUserWarning)
                merged_array_no_nan = convolve(merged_array_no_nan, gaussian_kernel)
            new_nan_pixel_sum = numpy.isnan(merged_array_no_nan[water_mask]).sum()
            pbar.update(nan_pixel_sum - new_nan_pixel_sum)
            nan_pixel_sum = new_nan_pixel_sum

    # Replace NaN values in the original array with the values from the convolved array
    merged_array_fixed = merged_array.copy()
    nan_pixel = numpy.isnan(merged_array_fixed)
    merged_array_fixed[water_mask & nan_pixel] = merged_array_no_nan[
        water_mask & nan_pixel
    ]

    return merged_array_fixed
