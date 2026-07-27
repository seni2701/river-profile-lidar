import geopandas
import pandas
from pathlib import Path
from shapely.geometry import LineString, Point
from tqdm import tqdm

from river_profile_lidar.bathymetry.trapezoid.cross_section_points.utils import (
    retrieve_polygon_for_pk,
    get_points_of_contact_between_two_polygon,
    farthest_points_from_list_of_points,
    get_boundaries_points_list,
    adjust_point_to_be_on_polygon_edge,
    get_extremities_points_from_points_before_and_after,
    create_all_points_from_shore_points_list,
)


def create_cross_section_points(data, folder_save_path, include_middle_point):
    """
    Create cross sections points from transect's polygon.
    Find intersects with touching polygon. Then interpolate points between shore and bottom.
    """
    pk_list = []
    error_list = []
    z_list = []
    points_list = []
    boundary_list = []
    line_list = []

    epsg = data.crs.to_epsg()
    for i in tqdm(range(1, len(data) - 1)):
        pk = data.loc[i]["PK"]
        transect_polygon = data.loc[i].geometry
        qi = data.loc[i]["Q_IMG_spli"]
        si = data.loc[i]["Pente"]
        wi = data.loc[i]["WAT_WIDTH"]
        if si <= 0 or pandas.isna(qi) or pandas.isna(si):
            continue

        polygon_before = retrieve_polygon_for_pk(data, pk - 5)
        polygon_after = retrieve_polygon_for_pk(data, pk + 5)

        intersect_points_before = get_points_of_contact_between_two_polygon(
            transect_polygon, polygon_before
        )
        intersect_points_after = get_points_of_contact_between_two_polygon(
            transect_polygon, polygon_after
        )

        if len(intersect_points_before) == 0 or len(intersect_points_after) == 0:
            continue

        # Something there's lot of touching points. We want the farthest apart
        intersect_points_before = farthest_points_from_list_of_points(
            intersect_points_before
        )
        intersect_points_after = farthest_points_from_list_of_points(
            intersect_points_after
        )

        # Add points on the upstream frontier
        intersect_points_after_point_class = [
            Point(point[0], point[1]) for point in intersect_points_after
        ]
        # From the the points on the shore. Create points along the line
        uptstream_middle_points_list = create_all_points_from_shore_points_list(
            intersect_points_after_point_class, qi, si, wi, include_middle_point
        )
        points_list.extend(uptstream_middle_points_list)
        pk_list.extend([pk] * len(uptstream_middle_points_list))
        z_list.extend([point.coords[0][2] for point in uptstream_middle_points_list])

        # Find the point that is in the middle of the transect
        boundary_list.extend(
            get_boundaries_points_list(intersect_points_before, intersect_points_after)
        )

        (
            int_point_list,
            int_line_list,
        ) = get_extremities_points_from_points_before_and_after(
            intersect_points_before, intersect_points_after
        )
        line_list.extend(int_line_list)

        adjusted_point_list = adjust_point_to_be_on_polygon_edge(
            int_point_list, transect_polygon
        )

        if LineString(adjusted_point_list).length == 0:
            error_list.append(i)
            continue

        all_middle_points_list = create_all_points_from_shore_points_list(
            adjusted_point_list, qi, si, wi, include_middle_point
        )
        points_list.extend(all_middle_points_list)
        pk_list.extend([pk] * len(all_middle_points_list))
        z_list.extend([point.coords[0][2] for point in all_middle_points_list])

    int_z_points_list = []
    for point in points_list:
        x, y, z = point.coords[0]
        int_z_points_list.append(Point(x, y, z))
    int_z_list = [z for z in z_list]
    df = pandas.DataFrame(
        {
            "PK": pk_list,
            "z": int_z_list,
        }
    )
    if include_middle_point is True:
        prefix = "trapezoid"
    else:
        prefix = "hab"

    geo_df = geopandas.GeoDataFrame(df, geometry=int_z_points_list, crs=f"EPSG:{epsg}")
    out_fp = Path(folder_save_path, f"{prefix}_cross_section_points.shp")
    out_fp.parent.mkdir(exist_ok=True, parents=True)
    geo_df.to_file(out_fp)
