import numpy as np


def flat_surface_distance(del_phi: float,
                          del_lambda: float,
                          phi_m: float) -> float:
    # Earth's radius in meters
    R = 6371000

    # convert to radians
    del_phi = np.deg2rad(del_phi)
    del_lambda = np.deg2rad(del_lambda)
    phi_m = np.deg2rad(phi_m)

    # calculate the distance
    d = R * np.sqrt(del_phi ** 2 + (np.cos(phi_m) * del_lambda) ** 2)

    return d


def GPS_to_coordinate_system(cam_nikita: np.ndarray,
                             cam_paul: np.ndarray,
                             midpoint: np.ndarray,
                             initial_point: np.ndarray,
                             point_of_impact: np.ndarray,
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # calculate the mean latitude
    phi_m = np.mean([cam_nikita[0], cam_paul[0], midpoint[0], initial_point[0], point_of_impact[0]])

    # place midpoit at (0, 0)
    cam_nikita = cam_nikita - midpoint
    cam_paul = cam_paul - midpoint
    initial_point = initial_point - midpoint
    point_of_impact = point_of_impact - midpoint
    midpoint = midpoint - midpoint

    # for each point,convert the coordinates to the flat surface (switching coordinates is intentional )

    cam_nikita = np.array([
        np.sign(cam_nikita[1]) * flat_surface_distance(0, cam_nikita[1], phi_m),
        np.sign(cam_nikita[0]) * flat_surface_distance(cam_nikita[0], 0, phi_m)
    ])

    cam_paul = np.array([
        np.sign(cam_paul[1]) * flat_surface_distance(0, cam_paul[1], phi_m),
        np.sign(cam_paul[0]) * flat_surface_distance(cam_paul[0], 0, phi_m)
    ])

    initial_point = np.array([
        np.sign(initial_point[1]) * flat_surface_distance(0, initial_point[1], phi_m),
        np.sign(initial_point[0]) * flat_surface_distance(initial_point[0], 0, phi_m)
    ])

    point_of_impact = np.array([
        np.sign(point_of_impact[1]) * flat_surface_distance(0, point_of_impact[1], phi_m),
        np.sign(point_of_impact[0]) * flat_surface_distance(point_of_impact[0], 0, phi_m)
    ])

    midpoint = np.array([
        0,
        0
    ])

    # caclulate rotational angle
    radius = np.sqrt(cam_nikita[0] ** 2 + cam_nikita[1] ** 2)

    cam_nikita_new = np.array([-radius, 0])

    angle = - np.arccos(np.dot(cam_nikita_new, cam_nikita) / (np.linalg.norm(cam_nikita) * np.linalg.norm(cam_nikita_new)))

    # calculate new coordimates for every vector

    cam_paul_new = np.array([
        cam_paul[0] * np.cos(angle) - cam_paul[1] * np.sin(angle),
        cam_paul[0] * np.sin(angle) + cam_paul[1] * np.cos(angle)
    ])

    initial_point_new = np.array([
        initial_point[0] * np.cos(angle) - initial_point[1] * np.sin(angle),
        initial_point[0] * np.sin(angle) + initial_point[1] * np.cos(angle)
    ])

    point_of_impact_new = np.array([
        point_of_impact[0] * np.cos(angle) - point_of_impact[1] * np.sin(angle),
        point_of_impact[0] * np.sin(angle) + point_of_impact[1] * np.cos(angle)
    ])

    midpoint_new = np.array([
        midpoint[0] * np.cos(angle) - midpoint[1] * np.sin(angle),
        midpoint[0] * np.sin(angle) + midpoint[1] * np.cos(angle)
    ])

    return cam_nikita_new, cam_paul_new, midpoint_new, initial_point_new, point_of_impact_new
