import os
import pickle

import numpy as np

from bcnf.simulation.camera import record_trajectory
from bcnf.simulation.physics import physics_ODE_simulation
from bcnf.simulation.sampling import get_cams_position
from bcnf.utils import get_dir


def render(data_name: str = 'data.pkl',
           n: int = 10,
           T: float = 2.0,
           FPS: int = 15,
           ratio: tuple = (16, 9),
           fov_horizontal: float = 70.0) -> dict:
    path = get_dir('data', 'bcnf-data')

    with open(os.path.join(path, data_name), 'rb') as f:
        data = pickle.load(f)

    render_dict: dict[str, list] = {
        'cams': [],
        'traj': []
    }

    # take the n first entries in data and put through physics_ODE_simulation and record_trajectory
    for i in range(n):
        x0_x = data['x0_x'][i]
        x0_y = data['x0_y'][i]
        x0_z = data['x0_z'][i]

        x0 = np.array([x0_x, x0_y, x0_z])

        v0_x = data['v0_x'][i]
        v0_y = data['v0_y'][i]
        v0_z = data['v0_z'][i]

        v0 = np.array([v0_x, v0_y, v0_z])

        w_x = data['w_x'][i]
        w_y = data['w_y'][i]
        w_z = data['w_z'][i]

        w = np.array([w_x, w_y, w_z])

        a_x = data['a_x'][i]
        a_y = data['a_y'][i]
        a_z = data['a_z'][i]

        a = np.array([a_x, a_y, a_z])

        g = data['g'][i]

        g = np.array([0, 0, g])

        rho = data['rho'][i]
        m = data['m'][i]
        b = data['b'][i]
        r = data['r'][i]

        traj = physics_ODE_simulation(x0, v0, g, w, b, m, rho, r, a, T, dt=(1 / FPS))

        cam_radius = data['cam_radius'][i]

        cam_radian = data['cam_radian'][i]
        cam_angles = data['cam_angles'][i]
        cam_heights = data['cam_heights'][i]

        cams = get_cams_position(cam_radian, cam_radius, cam_heights)

        recording = []

        for cam, angle in zip(cams, cam_angles):
            recording.append(record_trajectory(trajectory=traj, ratio=ratio, fov_horizontal=fov_horizontal, cam_pos=cam, make_gif=False, radius=r, viewing_angle=angle))

        render_dict['traj'].append(traj)
        render_dict['cams'].append(recording)

    return render_dict
