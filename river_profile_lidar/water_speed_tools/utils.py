from affine import Affine
import numpy
import pandas
from rasterstats import zonal_stats
from tqdm import tqdm


def get_vji(di, slope, y0):
    """
    Compute vji, the water velocity at each pixel, based on mean depth, slope, and y0.

    Parameters:
    di (numpy.ndarray or float):
        water depth at each pixel. A small constant (1e-8) is added to prevent numerical issues.
    slope (float):
        Channel slope.
    y0 (float):
        Height in the depth profile where flow velocity equals zero.

    Returns:
    numpy.ndarray or float:
        The computed water velocity at each pixel (vji), clipped to a minimum of 0.
    """
    di = di + 0.00000001
    vji = 2.5 * numpy.sqrt(9.81 * slope * di) * numpy.log(di / (numpy.exp(1) * y0))
    return numpy.clip(vji, 0, None)


def get_y0(di, v, slope):
    """
    Compute y0, the height in the velocity profile where flow velocity equals zero.

    Parameters:
    di (numpy.ma.MaskedArray): Array of depth from HAB (masked array to handle missing values).
    v (float): Mean flow velocity.
    slope (float): slope.

    Returns:
    numpy.ma.MaskedArray: The computed height y0.

    Notes:
    - A small constant (1e-8) is added to di in the logarithm to prevent undefined values.
    - Uses numpy.ma functions to handle masked (missing) data properly.
    """
    count_di = numpy.ma.count(di)
    di_sqrt = numpy.ma.sqrt(di)
    di_log = numpy.ma.log(di + 0.00000001)
    y0 = numpy.ma.exp(
        (
            numpy.ma.sum(di_sqrt * di_log)
            - (v * count_di) / (2.5 * numpy.ma.sqrt(9.81 * slope))
        )
        * (1 / (numpy.ma.sum(di_sqrt)))
    ) * (1 / numpy.ma.exp(1))
    return y0


def get_y0_from_transect_stats(di, v, slope):
    return get_y0(di=di, v=v, slope=slope)


def add_y0_to_transect_polygon_df(
    transect_polygon_df: pandas.DataFrame,
    hab_array: numpy.ndarray,
    transform: Affine,
) -> pandas.DataFrame:
    """
    Compute and add y0 (the height in the velocity profile where flow velocity equals zero)
    to a DataFrame containing transect polygon data.

    Parameters:
    transect_polygon_df (pandas.DataFrame):
        DataFrame containing transect polygons with geometry, velocity (Vi), and slope (Slope).
    hab_array (numpy.ndarray):
        3D array representing the HAB-derived data, where the first channel (hab_array[:, :, 0])
        contains depth values.
    transform (Affine):
        Affine transformation matrix for georeferencing the HAB array.

    Returns:
    pandas.DataFrame: The input DataFrame with an added "y0" column.
    """
    for i, row in tqdm(
        transect_polygon_df.iterrows(),
        total=len(transect_polygon_df),
        desc="Get Pixels by Transect",
    ):
        transect_polygon = row["geometry"]
        vi = row["Vi"]
        slope = row["Pente"]
        transect_stats = zonal_stats(
            transect_polygon,
            hab_array[:, :, 0],
            affine=transform,
            nodata=numpy.nan,
            add_stats={"y0": lambda x: get_y0_from_transect_stats(x, vi, slope)},
        )[0]
        transect_polygon_df.loc[i, "y0"] = transect_stats["y0"]
    return transect_polygon_df
