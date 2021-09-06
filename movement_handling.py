from numba import njit
from numpy import argwhere, sum, flatnonzero, sqrt, inf
import numpy as np
from scipy.spatial import cKDTree


def confine_particles(positions, v, x_max, y_max, r):
    max_boundaries = (x_max, y_max)
    for i in range(2):
        # Check if below lower limit
        is_outside = (((positions[:, i] < r).astype('intc') + (v[:, i] < 0).astype('intc')) == 2).astype('intc')
        outside_indices = argwhere(is_outside).flatten()
        positions[outside_indices, i] = r

        # Check if above upper limit
        is_outside = (((positions[:, i] > max_boundaries[i] - r).astype('intc') +
                       (v[:, i] > 0).astype('intc')) == 2).astype('intc')
        outside_indices = argwhere(is_outside).flatten()
        positions[outside_indices, i] = max_boundaries[i] - r


@njit
def get_differences_to_predator(position_pred, positions_prey):
    coordinate_differences = positions_prey - position_pred
    distances = sum(coordinate_differences ** 2, axis=1, keepdims=True)
    return np.append(coordinate_differences, distances, axis=1)


@njit
def get_distances_to_predator(position_pred, positions_prey):
    coordinate_differences = positions_prey - position_pred
    return sqrt(sum(coordinate_differences ** 2, axis=1))


@njit
def get_angle_to_predator(position_predator, positions_prey, v_predator):
    # arccos does exactly what I want here, returns the absolute value of the angle.
    pos_vector = positions_prey - position_predator
    cos = np.dot(pos_vector, v_predator) / (np.sqrt(np.sum(pos_vector**2, axis=1)) * np.sqrt(np.sum(v_predator**2)))
    return np.arccos(cos) / np.pi


@njit
def get_collision_indices(distances, r_pred, r_prey):
    return (np.arange(distances.size)[distances < (r_pred + r_prey)]).flatten()


def get_collision_indices_q_tree(positions, radius, limits):
    tree = cKDTree(positions, boxsize=limits)
    return tree.query_pairs(2*radius, p=2, output_type='ndarray')


@njit
def update_v(v, acc, speed):
    v += acc
    speeds = np.sqrt(np.sum(v**2, axis=1))
    above_speed_limit = flatnonzero(speeds > speed)
    for i in range(2):
        v[above_speed_limit, i] /= (speeds[above_speed_limit] / speed)


@njit
def move(positions, v):
    positions += v


@njit
def get_dead_indices(agents, int_max):
    return agents[:, 0][agents[:, 1] < int_max]
