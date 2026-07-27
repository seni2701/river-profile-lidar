import geopandas
from shapely.geometry import LineString, Point, MultiPolygon
import numpy
import itertools


POSSIBLE_COMBINAISONS = [[[0, 0], [1, 1]], [[0, 1], [1, 0]]]


def retrieve_polygon_for_pk(data, pk):
    polygon_list = data[data["PK"] == pk].geometry.values
    if len(polygon_list) == 1:
        return polygon_list[0]
    else:
        return MultiPolygon(polygon_list)


def extend_line(line, extend_factor):
    coords = list(line.coords)
    # Get the direction of the extension
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    # Extend the LineString by adding a point in the extension direction
    extended_x = coords[1][0] + dx * extend_factor
    extended_y = coords[1][1] + dy * extend_factor
    coords.append((extended_x, extended_y))
    extended_x = coords[0][0] - dx * extend_factor
    extended_y = coords[0][1] - dy * extend_factor
    coords.append((extended_x, extended_y))
    return LineString(coords)


def adjust_point_to_be_on_polygon_edge(point_list, polygon):
    adjusted_point_list = []
    extended_line_string = extend_line(LineString(point_list), 1)
    for point in point_list:
        adjusted_point = extended_line_string.intersection(polygon.boundary)
        if isinstance(adjusted_point, Point):
            adjusted_point = adjusted_point.xy
        else:
            adjusted_point = adjusted_point.geoms[
                numpy.argmin([point.distance(p) for p in adjusted_point.geoms])
            ].xy
        adjusted_point_list.append(
            geopandas.points_from_xy(adjusted_point[0], adjusted_point[1])[0]
        )
    return adjusted_point_list


def get_boundaries_points_list(intersect_points_before, intersect_points_after):
    boundaries_points_list = []
    boundaries_points_list.append(
        geopandas.points_from_xy(
            [intersect_points_before[0][0]], [intersect_points_before[0][1]]
        )[0]
    )
    boundaries_points_list.append(
        geopandas.points_from_xy(
            [intersect_points_before[1][0]], [intersect_points_before[1][1]]
        )[0]
    )
    boundaries_points_list.append(
        geopandas.points_from_xy(
            [intersect_points_after[0][0]], [intersect_points_after[0][1]]
        )[0]
    )
    boundaries_points_list.append(
        geopandas.points_from_xy(
            [intersect_points_after[1][0]], [intersect_points_after[1][1]]
        )[0]
    )
    return boundaries_points_list


def get_points_of_contact_between_two_polygon(polygon1, polygon2):
    # Find common boundary
    common_boundary = polygon1.intersection(polygon2)

    # Get points of contact (vertices on common boundary)
    points_of_contact = []
    if common_boundary.geom_type == "LineString":
        points_of_contact.append(common_boundary.coords[0])
        points_of_contact.append(common_boundary.coords[-1])
    elif common_boundary.geom_type == "MultiLineString":
        for line in common_boundary.geoms:
            points_of_contact.append(line.coords[0])
            points_of_contact.append(line.coords[-1])
    return numpy.array(points_of_contact)


def farthest_points_from_list_of_points(list_of_points):
    list_of_points = geopandas.points_from_xy(
        list_of_points[:, 0], list_of_points[:, 1]
    )
    max_distance = 0
    farthest_pair = None

    # Iterate over all pairs of points and calculate distance
    for pair in itertools.combinations(list_of_points, 2):
        dist = pair[0].distance(pair[1])
        if dist > max_distance:
            max_distance = dist
            farthest_pair = pair
    points_array = numpy.array(
        [[point.xy[0][0], point.xy[1][0]] for point in farthest_pair]
    )
    return points_array


def get_extremities_points_from_points_before_and_after(
    intersect_points_before, intersect_points_after
):
    """
    With the points before and after, find the points in the middle of the transect with the middle of a line.
    Need to select the good two points
    """
    points_list = []
    valid = False
    for points_combinaisons_1, points_combinaisons_2 in POSSIBLE_COMBINAISONS:
        first_line_string = LineString(
            [
                intersect_points_before[points_combinaisons_1[0]],
                intersect_points_after[points_combinaisons_1[1]],
            ]
        )
        second_line_string = LineString(
            [
                intersect_points_before[points_combinaisons_2[0]],
                intersect_points_after[points_combinaisons_2[1]],
            ]
        )
        if first_line_string.intersects(second_line_string) is False:
            valid = True
            line_list = [first_line_string, second_line_string]
            points_list.append(
                LineString(
                    [
                        intersect_points_before[points_combinaisons_1[0]],
                        intersect_points_after[points_combinaisons_1[1]],
                    ]
                ).centroid
            )
            points_list.append(
                LineString(
                    [
                        intersect_points_before[points_combinaisons_2[0]],
                        intersect_points_after[points_combinaisons_2[1]],
                    ]
                ).centroid
            )
            break
    if valid is False:
        raise ValueError
    return points_list, line_list


def add_z_to_points_list_from_z_list(points_list, z_list):
    new_points_list = []
    for point, z in zip(points_list, z_list):
        x, y = point.xy
        new_points_list.append(Point(x[0], y[0], z))
    return new_points_list


def get_middle_point_from_two_points(point1, point2):
    line = LineString([point1, point2])
    x, y = line.centroid.coords[0]
    z = (point1.coords[0][2] + point2.coords[0][2]) / 2
    return [Point(x, y, z)]


def create_all_points_from_shore_points_list(
    shore_points_list, qi, si, wi, include_middle_point
):
    points_list = []
    shore_points_z = add_z_to_points_list_from_z_list(shore_points_list, [0] * 2)
    points_list.extend(shore_points_z)

    line_string = LineString(shore_points_list)
    bottom_points = [
        line_string.line_interpolate_point(0.1, normalized=True),
        line_string.line_interpolate_point(0.9, normalized=True),
    ]

    di = (qi / (3.125 * wi * (si**0.12))) ** 0.55
    hi = 2 * di / 1.8
    bottom_points_z = add_z_to_points_list_from_z_list(bottom_points, [hi] * 2)
    points_list.extend(bottom_points_z)

    # Add point between shore and the bottom
    for shore_points in shore_points_z:
        # Starting with one point on the shore, finding the closest point in the bottom.
        # Interpolate a point between those two
        corresponding_bottom_points = bottom_points_z[
            numpy.argmin(
                [
                    shore_points.distance(bottom_points)
                    for bottom_points in bottom_points_z
                ]
            )
        ]
        shore_bottom_points_interpolation_list = get_middle_point_from_two_points(
            shore_points, corresponding_bottom_points
        )
        points_list.extend(shore_bottom_points_interpolation_list)

    # Add point between bottom points
    bottom_points_interpolation_list = get_middle_point_from_two_points(
        bottom_points_z[0], bottom_points_z[1]
    )
    if include_middle_point:
        points_list.extend(bottom_points_interpolation_list)
    return points_list
