import os
import pickle

import numpy as np
from dynaconf import Dynaconf
from tqdm import tqdm

from bcnf.simulation.camera import record_trajectory
from bcnf.simulation.physics import calculate_point_of_impact, physics_ODE_simulation
from bcnf.utils import get_dir


def sample_from_config(values: dict
                       ) -> float:
    distribution_type = values['distribution']
    if distribution_type == 'uniform':
        if "min" not in values or "max" not in values:
            raise ValueError('min and max must be defined for uniform distribution')
        return np.random.uniform(0, 1)
    elif distribution_type == 'gaussian':
        return np.random.normal(0, 1)  # mean and std are used as parameters for the standard normal distribution later
    elif distribution_type == 'gamma':
        if "shape" not in values or "scale" not in values:
            raise ValueError('shape and scale must be defined for gamma distribution')
        return np.random.gamma(values['shape'], values['scale'])
    else:
        raise ValueError(f'Unknown distribution type: {distribution_type}')


def get_cams_position(cam_radiants: np.ndarray = np.array([0, 0]),
                      cam_circle_radius: float = 25,
                      cam_heights: np.ndarray = np.array([1, 1])
                      ) -> list[np.ndarray]:
    cams = []
    for cam_radiant, cam_height in (cam_radiants, cam_heights):
        cams.append(np.array([-cam_circle_radius * np.cos(cam_radiant), cam_circle_radius * np.sin(cam_radiant), cam_height]))
    return cams


def accept_visibility(cams: list[np.ndarray]) -> bool:
    vis = np.sum([np.sum(cam) / len(cam) for cam in cams]) / len(cams)

    if vis > 0.75:
        return True
    elif (1 / (1 + np.exp(-(vis - 0.5) * 10))) / 1.5 > np.random.uniform(0, 1):  # modified sigmoid
        return True
    else:
        return False


def accept_traveled_distance(distance: float,
                             cam_radius: float) -> bool:
    # assuming uniform distance distribution between 0 m and cam_radius: acceptance rate of ~70 %
    ratio = distance / cam_radius
    if ratio > 0.75:
        return True
    elif np.sqrt(ratio) > np.random.uniform(0, 1):  # square root acceptance
        return True
    else:
        return False


def sample_ballistic_parameters(num_cams: int = 2,
                                cfg_file: str = f'{get_dir()}/configs/config.yaml'
                                ) -> tuple:
    config = Dynaconf(settings_files=[cfg_file])

    # pos
    if config['x0']['x0_xy']['distribution'] == 'gaussian':
        r_x = np.sqrt(np.abs(sample_from_config(config['x0']['x0_xy']))) * config['x0']['x0_xy']['std'] + config['x0']['x0_xy']['mean']
    elif config['x0']['x0_xy']['distribution'] == 'uniform':
        r_x = np.sqrt(sample_from_config(config['x0']['x0_xy'])) * (config['x0']['x0_xy']['max'] - config['x0']['x0_xy']['min']) + config['x0']['x0_xy']['min']

    phi = np.random.uniform(0, 2 * np.pi)

    x0_x = r_x * np.cos(phi)
    x0_y = r_x * np.sin(phi)

    if config['x0']['x0_z']['distribution'] == 'gaussian':
        x0_z = sample_from_config(config['x0']['x0_z']) * config['x0']['x0_z']['std'] + config['x0']['x0_z']['mean']
    elif config['x0']['x0_z']['distribution'] == 'uniform':
        x0_z = sample_from_config(config['x0']['x0_z']) * (config['x0']['x0_z']['max'] - config['x0']['x0_z']['min']) + config['x0']['x0_z']['min']

    x0 = np.array([x0_x, x0_y, x0_z])

    # velo
    if config['v0']['v0_xy']['distribution'] == 'gaussian':
        r_v = np.sqrt(np.abs(sample_from_config(config['v0']['v0_xy']))) * config['v0']['v0_xy']['std'] + config['v0']['v0_xy']['mean']
    elif config['v0']['v0_xy']['distribution'] == 'uniform':
        r_v = np.sqrt(sample_from_config(config['v0']['v0_xy'])) * (config['v0']['v0_xy']['max'] - config['v0']['v0_xy']['min']) + config['v0']['v0_xy']['min']

    phi_v = np.random.uniform(0, 2 * np.pi)

    v0_x = r_v * np.cos(phi_v)
    v0_y = r_v * np.sin(phi_v)

    if config['v0']['v0_z']['distribution'] == 'gaussian':
        v0_z = sample_from_config(config['v0']['v0_z']) * config['v0']['v0_z']['std'] + config['v0']['v0_z']['mean']
    elif config['v0']['v0_z']['distribution'] == 'uniform':
        v0_z = sample_from_config(config['v0']['v0_z']) * (config['v0']['v0_z']['max'] - config['v0']['v0_z']['min']) + config['v0']['v0_z']['min']

    v0 = np.array([v0_x, v0_y, v0_z])

    # wind
    if config['w']['w_xy']['distribution'] == 'gaussian':
        r_x = np.sqrt(np.abs(sample_from_config(config['w']['w_xy']))) * config['w']['w_xy']['std'] + config['w']['w_xy']['mean']
    elif config['w']['w_xy']['distribution'] == 'uniform':
        r_x = np.sqrt(sample_from_config(config['w']['w_xy'])) * (config['w']['w_xy']['max'] - config['w']['w_xy']['min']) + config['w']['w_xy']['min']

    phi_w = np.random.uniform(0, 2 * np.pi)

    w_x = r_x * np.cos(phi_w)
    w_y = r_x * np.sin(phi_w)

    if config['w']['w_z']['distribution'] == 'gaussian':
        w_z = sample_from_config(config['w']['w_z']) * config['w']['w_z']['std'] + config['w']['w_z']['mean']
    elif config['w']['w_z']['distribution'] == 'uniform':
        w_z = sample_from_config(config['w']['w_z']) * (config['w']['w_z']['max'] - config['w']['w_z']['min']) + config['w']['w_z']['min']

    w = np.array([w_x, w_y, w_z])

    # thrust

    if config['a']['distribution'] == 'gaussian':
        r_a = np.cbrt(np.abs(sample_from_config(config['a']))) * config['a']['std'] + config['a']['mean']
    elif config['a']['distribution'] == 'uniform':
        r_a = np.cbrt(sample_from_config(config['a'])) * (config['a']['max'] - config['a']['min']) + config['a']['min']

    phi_a = np.random.uniform(0, 2 * np.pi)
    theta_a = np.random.uniform(0, np.pi)

    a_x = r_a * np.sin(theta_a) * np.cos(phi_a)
    a_y = r_a * np.sin(theta_a) * np.sin(phi_a)
    a_z = r_a * np.cos(theta_a)

    a = np.array([a_x, a_y, a_z])

    # grav
    if config['g']['distribution'] == 'gamma':
        g_z = sample_from_config(config['g'])
    elif config['g']['distribution'] == 'uniform':
        g_z = sample_from_config(config['g']) * (config['g']['max'] - config['g']['min']) + config['g']['min']

    g = np.array([0, 0, -g_z])

    # b
    # density of atmosphere
    if config['rho']['distribution'] == 'gamma':
        rho = sample_from_config(config['rho'])
    elif config['rho']['distribution'] == 'uniform':
        rho = sample_from_config(config['rho']) * (config['rho']['max'] - config['rho']['min']) + config['rho']['min']

    # radius of ball
    if config['r_ball']['distribution'] == 'gamma':
        r = sample_from_config(config['r_ball'])
    elif config['r_ball']['distribution'] == 'uniform':
        r = sample_from_config(config['r_ball']) * (config['r_ball']['max'] - config['r_ball']['min']) + config['r_ball']['min']

    # area of thown object
    A = np.pi * r**2

    # drag coefficient
    if config['Cd']['distribution'] == 'gamma':
        Cd = sample_from_config(config['Cd'])
    elif config['Cd']['distribution'] == 'uniform':
        Cd = sample_from_config(config['Cd']) * (config['Cd']['max'] - config['Cd']['min']) + config['Cd']['min']

    b = rho * A * Cd

    # mass
    if config['m']['distribution'] == 'gamma':
        m = sample_from_config(config['m'])
    elif config['m']['distribution'] == 'uniform':
        m = sample_from_config(config['m']) * (config['m']['max'] - config['m']['min']) + config['m']['min']

    # second cam position
    cam_radian_array = [(sample_from_config(config['cam_radian']) * (config['cam_radian']['max'] - config['cam_radian']['min']) + config['cam_radian']['min']) for _ in range(num_cams - 1)]

    # cam radius
    if config['cam_radius']['distribution'] == 'gamma':
        cam_radius = sample_from_config(config['cam_radius'])
    elif config['cam_radius']['distribution'] == 'uniform':
        cam_radius = sample_from_config(config['cam_radius']) * (config['cam_radius']['max'] - config['cam_radius']['min']) + config['cam_radius']['min']

    # cam angles
    if config['cam_angle']['distribution'] == 'gamma':
        cam_angles = [sample_from_config(config['cam_angle']) for _ in range(num_cams)]
    elif config['cam_angle']['distribution'] == 'uniform':
        cam_angles = [sample_from_config(config['cam_angle']) * (config['cam_angle']['max'] - config['cam_angle']['min']) + config['cam_angle']['min'] for _ in range(num_cams)]

    # cam heights
    cam_heights = [sample_from_config(config['cam_heights']) * (config['cam_heights']['max'] - config['cam_heights']['min']) + config['cam_heights']['min'] for _ in range(num_cams)]

    return x0, v0, g, w, b, m, a, cam_radian_array, r, A, Cd, rho, cam_radius, cam_angles, cam_heights


def generate_data(
        n: int = 100,
        type: str = 'parameters',  # 'render', 'trajectory', or 'parameters'
        SPF: float = 1 / 30,
        T: float = 4,
        ratio: tuple = (16, 9),
        fov_horizontal: float = 70.0,
        cam1_radian: float = 0.0,
        print_acc_rej: bool = False,
        name: str | None = None,
        num_cams: int = 2,
        config_file: str = f'{get_dir()}/configs/config.yaml',
        break_on_impact: bool = True,
        verbose: bool = False) -> dict[str, list]:

    accepted_count = 0
    rejected_count = 0

    data: dict[str, list] = {
        'cams': [],
        'traj': [],
        'x0_x': [],
        'x0_y': [],
        'x0_z': [],
        'v0_x': [],
        'v0_y': [],
        'v0_z': [],
        'g': [],
        'w_x': [],
        'w_y': [],
        'w_z': [],
        'b': [],
        'A': [],
        'Cd': [],
        'rho': [],
        'm': [],
        'a_x': [],
        'a_y': [],
        'a_z': [],
        'cam_radian': [],
        'r': [],
        'cam_radius': [],
        'cam_angles': [],
        'cam_heights': []
    }

    pbar = tqdm(total=n, disable=not verbose)

    while accepted_count < n:
        x0, v0, g, w, b, m, a, cam_radian_array, r, A, Cd, rho, cam_radius, cam_angles, cam_heights = sample_ballistic_parameters(num_cams=num_cams, cfg_file=config_file)

        # first check: will the ball actually come down again?
        if g[2] + a[2] > 0:
            rejected_count += 1
            pbar.set_postfix(accepted=accepted_count, rejected=rejected_count, ratio=accepted_count / (accepted_count + rejected_count))
            continue

        # second check: is x0_z > 0?
        if x0[2] < 0:
            rejected_count += 1
            pbar.set_postfix(accepted=accepted_count, rejected=rejected_count, ratio=accepted_count / (accepted_count + rejected_count))
            continue

        # third check: how far does the ball travel?
        poi = calculate_point_of_impact(x0, v0, g, w, b, m, rho, r, a)

        # Check that poi and x0 have the same shape
        if poi.shape != x0.shape:
            raise ValueError(f'poi and x0 must have the same shape. poi.shape = {poi.shape}, x0.shape = {x0.shape}')
        distance = float(np.linalg.norm(poi - x0))

        if not accept_traveled_distance(distance, cam_radius):
            rejected_count += 1
            pbar.set_postfix(accepted=accepted_count, rejected=rejected_count, ratio=accepted_count / (accepted_count + rejected_count))
            continue

        traj = physics_ODE_simulation(x0, v0, g, w, b, m, rho, r, a, T, SPF, break_on_impact=break_on_impact)

        # Prepend the first camera radian to the other camera radians
        cam_radian_array = np.insert(cam_radian_array, 0, cam1_radian)
        cams_pos = get_cams_position(cam_radian_array, cam_radius, cam_heights)

        cams = []
        for cam, angle in zip(cams_pos, cam_angles):
            cams.append(record_trajectory(traj, ratio, fov_horizontal, cam, make_gif=False, radius=r, viewing_angle=angle))

        if not accept_visibility(cams):
            rejected_count += 1
            pbar.set_postfix(accepted=accepted_count, rejected=rejected_count, ratio=accepted_count / (accepted_count + rejected_count))
            continue

        # add to list
        if type == 'render':
            # append cam1, cam2 and parameters
            data['cams'].append(cams)
            data['traj'].append(traj)
            data['x0_x'].append(x0[0])
            data['x0_y'].append(x0[1])
            data['x0_z'].append(x0[2])
            data['v0_x'].append(v0[0])
            data['v0_y'].append(v0[1])
            data['v0_z'].append(v0[2])
            data['g'].append(g[2])
            data['w_x'].append(w[0])
            data['w_y'].append(w[1])
            data['w_z'].append(w[2])
            data['b'].append(b)
            data['A'].append(A)
            data['Cd'].append(Cd)
            data['rho'].append(rho)
            data['m'].append(m)
            data['a_x'].append(a[0])
            data['a_y'].append(a[1])
            data['a_z'].append(a[2])
            data['cam_radian'].append(cam_radian_array)
            data['r'].append(r)
            data['cam_radius'].append(cam_radius)
            data['cam_angles'].append(cam_angles)
            data['cam_heights'].append(cam_heights)

        elif type == 'parameters':
            data['x0_x'].append(x0[0])
            data['x0_y'].append(x0[1])
            data['x0_z'].append(x0[2])
            data['v0_x'].append(v0[0])
            data['v0_y'].append(v0[1])
            data['v0_z'].append(v0[2])
            data['g'].append(g[2])
            data['w_x'].append(w[0])
            data['w_y'].append(w[1])
            data['w_z'].append(w[2])
            data['b'].append(b)
            data['A'].append(A)
            data['Cd'].append(Cd)
            data['rho'].append(rho)
            data['m'].append(m)
            data['a_x'].append(a[0])
            data['a_y'].append(a[1])
            data['a_z'].append(a[2])
            data['cam_radian'].append(cam_radian_array)
            data['r'].append(r)
            data['cam_radius'].append(cam_radius)
            data['cam_angles'].append(cam_angles)
            data['cam_heights'].append(cam_heights)

        elif type == 'trajectory':
            data['traj'].append(traj)
            data['x0_x'].append(x0[0])
            data['x0_y'].append(x0[1])
            data['x0_z'].append(x0[2])
            data['v0_x'].append(v0[0])
            data['v0_y'].append(v0[1])
            data['v0_z'].append(v0[2])
            data['g'].append(g[2])
            data['w_x'].append(w[0])
            data['w_y'].append(w[1])
            data['w_z'].append(w[2])
            data['b'].append(b)
            data['A'].append(A)
            data['Cd'].append(Cd)
            data['rho'].append(rho)
            data['m'].append(m)
            data['a_x'].append(a[0])
            data['a_y'].append(a[1])
            data['a_z'].append(a[2])
            data['cam_radian'].append(cam_radian_array)
            data['r'].append(r)
            data['cam_radius'].append(cam_radius)
            data['cam_angles'].append(cam_angles)
            data['cam_heights'].append(cam_heights)
        else:
            raise ValueError('type must be one of "render", "trajectory", or "parameters"')

        accepted_count += 1

        if accepted_count % 100 == 0 and name is not None:
            with open(os.path.join(get_dir('data', 'bcnf-data', create=True), name + '.pkl'), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        # print(f'accepted: {accepted_count}, rejected: {rejected_count}', end='\r')
        pbar.update(1)
        if print_acc_rej:
            pbar.set_postfix(accepted=accepted_count, rejected=rejected_count, ratio=accepted_count / (accepted_count + rejected_count))

    pbar.close()

    if name is not None:
        with open(os.path.join(get_dir('data', 'bcnf-data', create=True), name + '.pkl'), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data
