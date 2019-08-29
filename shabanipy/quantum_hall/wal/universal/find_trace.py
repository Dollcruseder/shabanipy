# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to compute the trace of evolution matrix

"""
from math import cos, exp, pi

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def find_trace(l: np.ndarray, c_phi: np.ndarray, c_3phi: np.ndarray, s_phi: np.ndarray, s_3phi: np.ndarray,
               alpha: float, beta3: float,beta1: float, k: float, hvf: float) -> float:
    """Find the trace of the matrix R_tot^2

    Parameters
    ----------
    l: np.ndarray
        (n_scat) array, length of each segment
    angle: np.ndarray
        (n_scat) array, angle of each segment
    alpha: float
        Rashba SOI coeffecient
    beta3: float
        Cubic Dresselhaus coeffecient
    beta1: float
        Linear Dresselhaus coeffecient
    k: float

    hvf: float


    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2

    """
    # XXX all of this could be moved outside of this function this the
    # allocations may not play well with prange
    rotations = np.empty((len(l), 2, 2), dtype=np.complex128)

    B_x = alpha * k * s_phi + beta3 * (k ** 3) * c_3phi + beta1 * k * c_phi
    B_y = -alpha * k * c_phi + beta3 * (k ** 3) * s_3phi - beta1 * k * s_phi
    B = np.sqrt(B_x ** 2 + B_y ** 2)
    theta = 2 * B * l / hvf
    c_theta = np.cos(0.5*theta)
    s_theta = np.sin(0.5*theta)

    psi1 = np.empty(len(l), dtype=np.complex128)
    psi2 = np.empty(len(l), dtype=np.complex128)
    for i, (b, bx, by) in enumerate(zip(B, B_x, B_y)):
        if b != 0:
            psi1[i] = -1j * (bx / b + 1j * by /b)
            psi2[i] = -1j * (bx / b - 1j * by /b)
        else:
            psi1[i] = psi2[i] = 0

    rotations[:, 0, 0] = c_theta
    rotations[:, 0, 1] = psi1 * s_theta
    rotations[:, 1, 0] = psi2 * s_theta
    rotations[:, 1, 1] = c_theta

    # For 2x2 matrices calling BLAS matrix multiplication has a large overhead
    # and the need to allocate the output matrix is likely to cause issue with
    # parallelization of the code.
    cw_rot = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    for i in range(0, len(l)):
        # equivalent to cw_rot = r @ cw_rot
        r = rotations[i]
        a = r[0, 0]*cw_rot[0, 0] + r[0, 1]*cw_rot[1,0]
        b = r[0, 0]*cw_rot[0, 1] + r[0, 1]*cw_rot[1,1]
        c = r[1, 0]*cw_rot[0, 0] + r[1, 1]*cw_rot[1,0]
        d = r[1, 0]*cw_rot[0, 1] + r[1, 1]*cw_rot[1,1]
        cw_rot[0, 0] = a
        cw_rot[0, 1] = b
        cw_rot[1, 0] = c
        cw_rot[1, 1] = d

    return (cw_rot[0, 0]*cw_rot[0, 0] +
            cw_rot[0, 1]*cw_rot[1, 0] +
            cw_rot[1, 0]*cw_rot[0, 1] +
            cw_rot[1, 1]*cw_rot[1, 1]).real



@njit(fastmath=True, parallel=True)
def compute_traces(index, l, c_phi, c_3phi, s_phi, s_3phi, alpha, beta1, beta3, k, hvf, N_orbit):
    """Compute the trace of the evolution operator for different trajectories

    Parameters
    ----------
    index : [type]
        [description]
    l : [type]
        [description]
    angle : [type]
        [description]
    alpha : [type]
        [description]
    beta1 : [type]
        [description]
    beta3 : [type]
        [description]
    k : [type]
        [description]
    hvf : [type]
        [description]
    N_orbit : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    """
    T = np.empty(N_orbit)


    for n in prange(N_orbit//1000):
        for i in range(1000):
            traj_id = n*1000 + i
            if traj_id >= N_orbit:
                break
            begin, end = index[traj_id]
            T_a = find_trace(l[begin:end], c_phi[begin:end], c_3phi[begin:end], s_phi[begin:end], s_3phi[begin:end],
                             alpha, beta3, beta1, k, hvf)
            T[traj_id] = T_a


    return T

@njit(fastmath=True)
def find_trace_theta(l: np.ndarray, c_phi: np.ndarray, c_3phi: np.ndarray, s_phi: np.ndarray, s_3phi: np.ndarray,
                     theta_alpha: float, theta_beta3: float, theta_beta1: float) -> float:
    """Find the trace of the matrix R_tot^2

    Parameters
    ----------
    l: np.ndarray
        (n_scat) array, length of each segment
    angle: np.ndarray
        (n_scat) array, angle of each segment
    theta_alpha: float
        Rashba SOI coeffecient
    theta_beta3: float
        Cubic Dresselhaus coeffecient
    theta_beta1: float
        Linear Dresselhaus coeffecient

    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2

    """
    # XXX all of this could be moved outside of this function this the
    # allocations may not play well with prange
    rotations = np.empty((len(l), 2, 2), dtype=np.complex128)

    B_x = theta_alpha * s_phi + theta_beta3 * c_3phi + theta_beta1 * c_phi
    B_y = -theta_alpha * c_phi + theta_beta3 * s_3phi - theta_beta1 * s_phi
    B = np.sqrt(B_x ** 2 + B_y ** 2)
    theta = B * l
    c_theta = np.cos(0.5*theta)
    s_theta = np.sin(0.5*theta)

    psi1 = np.empty(len(l), dtype=np.complex128)
    psi2 = np.empty(len(l), dtype=np.complex128)
    for i, (b, bx, by) in enumerate(zip(B, B_x, B_y)):
        if b != 0:
            psi1[i] = -1j * (bx / b + 1j * by /b)
            psi2[i] = -1j * (bx / b - 1j * by /b)
        else:
            psi1[i] = psi2[i] = 0

    rotations[:, 0, 0] = c_theta
    rotations[:, 0, 1] = psi1 * s_theta
    rotations[:, 1, 0] = psi2 * s_theta
    rotations[:, 1, 1] = c_theta

    # For 2x2 matrices calling BLAS matrix multiplication has a large overhead
    # and the need to allocate the output matrix is likely to cause issue with
    # parallelization of the code.
    cw_rot = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    for i in range(0, len(l)):
        # equivalent to cw_rot = r @ cw_rot
        r = rotations[i]
        a = r[0, 0]*cw_rot[0, 0] + r[0, 1]*cw_rot[1,0]
        b = r[0, 0]*cw_rot[0, 1] + r[0, 1]*cw_rot[1,1]
        c = r[1, 0]*cw_rot[0, 0] + r[1, 1]*cw_rot[1,0]
        d = r[1, 0]*cw_rot[0, 1] + r[1, 1]*cw_rot[1,1]
        cw_rot[0, 0] = a
        cw_rot[0, 1] = b
        cw_rot[1, 0] = c
        cw_rot[1, 1] = d

    return (cw_rot[0, 0]*cw_rot[0, 0] +
            cw_rot[0, 1]*cw_rot[1, 0] +
            cw_rot[1, 0]*cw_rot[0, 1] +
            cw_rot[1, 1]*cw_rot[1, 1]).real



@njit(fastmath=True, parallel=True)
def compute_traces_theta(index, l, c_phi, c_3phi, s_phi, s_3phi, theta_alpha, theta_beta1, theta_beta3, N_orbit):
    """Compute the trace of the evolution operator for different trajectories

    Parameters
    ----------
    index : [type]
        [description]
    l : [type]
        [description]
    angle : [type]
        [description]
    theta_alpha: float
        Rashba SOI coeffecient
    theta_beta3: float
        Cubic Dresselhaus coeffecient
    theta_beta1: float
        Linear Dresselhaus coeffecient
    N_orbit : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    """
    T = np.empty(N_orbit)


    for n in prange(N_orbit//1000):
        for i in range(1000):
            traj_id = n*1000 + i
            if traj_id >= N_orbit:
                break
            begin, end = index[traj_id]
            T_a = find_trace_theta(l[begin:end], c_phi[begin:end], c_3phi[begin:end], s_phi[begin:end], s_3phi[begin:end],
                                   theta_alpha, theta_beta3, theta_beta1)
            T[traj_id] = T_a


    return T


@njit(fastmath=True, parallel=True)
def find_trace_Zeeman(l: np.ndarray, c_phi: np.ndarray, c_3phi: np.ndarray, s_phi: np.ndarray, s_3phi: np.ndarray,
                      theta_alpha: float, theta_beta3: float, theta_beta1: float,
                      B_zx: float, B_zy: float) -> float:
    """Find the trace of the matrix R_tot^2

    Parameters
    ----------
    l: np.ndarray
        (n_scat) array, length of each segment
    angle: np.ndarray
        (n_scat) array, angle of each segment
    theta_alpha: float
        Rashba SOI coeffecient
    theta_beta3: float
        Cubic Dresselhaus coeffecient
    theta_beta1: float
        Linear Dresselhaus coeffecient

    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2

    """
    # XXX all of this could be moved outside of this function this the
    # allocations may not play well with prange
    rotations_cw = np.empty((len(l), 2, 2), dtype=np.complex128)

    B_x_cw = theta_alpha * s_phi + theta_beta3 * c_3phi + theta_beta1 * c_phi + B_zx
    B_y_cw = -theta_alpha * c_phi + theta_beta3 * s_3phi - theta_beta1 * s_phi + B_zy
    B_cw = np.sqrt(B_x_cw ** 2 + B_y_cw ** 2)
    theta_cw = B_cw * l
    c_theta_cw = np.cos(0.5*theta_cw)
    s_theta_cw = np.sin(0.5*theta_cw)

    psi1_cw = np.empty(len(l), dtype=np.complex128)
    psi2_cw = np.empty(len(l), dtype=np.complex128)
    for i, (b, bx, by) in enumerate(zip(B_cw, B_x_cw, B_y_cw)):
        if b != 0:
            psi1_cw[i] = -1j * (bx / b + 1j * by /b)
            psi2_cw[i] = -1j * (bx / b - 1j * by /b)
        else:
            psi1_cw[i] = psi2_cw[i] = 0

    rotations_cw[:, 0, 0] = c_theta_cw
    rotations_cw[:, 0, 1] = psi1_cw * s_theta_cw
    rotations_cw[:, 1, 0] = psi2_cw * s_theta_cw
    rotations_cw[:, 1, 1] = c_theta_cw

    # For 2x2 matrices calling BLAS matrix multiplication has a large overhead
    # and the need to allocate the output matrix is likely to cause issue with
    # parallelization of the code.
    cw_rot = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    for i in range(0, len(l)):
        # equivalent to cw_rot = r @ cw_rot
        r = rotations_cw[i]
        a = r[0, 0]*cw_rot[0, 0] + r[0, 1]*cw_rot[1,0]
        b = r[0, 0]*cw_rot[0, 1] + r[0, 1]*cw_rot[1,1]
        c = r[1, 0]*cw_rot[0, 0] + r[1, 1]*cw_rot[1,0]
        d = r[1, 0]*cw_rot[0, 1] + r[1, 1]*cw_rot[1,1]
        cw_rot[0, 0] = a
        cw_rot[0, 1] = b
        cw_rot[1, 0] = c
        cw_rot[1, 1] = d


    rotations_ccw = np.empty((len(l), 2, 2), dtype=np.complex128)

    B_x_ccw = - theta_alpha * s_phi - theta_beta3 * c_3phi - theta_beta1 * c_phi + B_zx
    B_y_ccw = theta_alpha * c_phi - theta_beta3 * s_3phi + theta_beta1 * s_phi + B_zy
    B_ccw = np.sqrt(B_x_ccw ** 2 + B_y_ccw ** 2)
    theta_ccw = B_ccw * l
    c_theta_ccw = np.cos(0.5*theta_ccw)
    s_theta_ccw = np.sin(0.5*theta_ccw)

    psi1_ccw = np.empty(len(l), dtype=np.complex128)
    psi2_ccw = np.empty(len(l), dtype=np.complex128)
    for i, (b, bx, by) in enumerate(zip(B_ccw, B_x_ccw, B_y_ccw)):
        if b != 0:
            psi1_ccw[i] = -1j * (bx / b + 1j * by /b)
            psi2_ccw[i] = -1j * (bx / b - 1j * by /b)
        else:
            psi1_ccw[i] = psi2_ccw[i] = 0

    rotations_ccw[:, 0, 0] = c_theta_ccw
    rotations_ccw[:, 0, 1] = psi1_ccw * s_theta_ccw
    rotations_ccw[:, 1, 0] = psi2_ccw * s_theta_ccw
    rotations_ccw[:, 1, 1] = c_theta_ccw
    ccw_rot = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    for i in range(len(l)-1, -1, -1):
        # equivalent to ccw_rot = r @ ccw_rot
        r = rotations_ccw[i]
        a = r[0, 0]*ccw_rot[0, 0] + r[0, 1]*ccw_rot[1,0]
        b = r[0, 0]*ccw_rot[0, 1] + r[0, 1]*ccw_rot[1,1]
        c = r[1, 0]*ccw_rot[0, 0] + r[1, 1]*ccw_rot[1,0]
        d = r[1, 0]*ccw_rot[0, 1] + r[1, 1]*ccw_rot[1,1]
        ccw_rot[0, 0] = a
        ccw_rot[0, 1] = b
        ccw_rot[1, 0] = c
        ccw_rot[1, 1] = d

    return (ccw_rot[0, 0].conjugate()*cw_rot[0, 0] +
            ccw_rot[1, 0].conjugate()*cw_rot[1, 0] +
            ccw_rot[0, 1].conjugate()*cw_rot[0, 1] +
            ccw_rot[1, 1].conjugate()*cw_rot[1, 1]).real


@njit(fastmath=True, parallel=True)
def compute_traces_Zeeman(index, l, c_phi, c_3phi, s_phi, s_3phi, theta_alpha, theta_beta1, theta_beta3, N_orbit, B_zx, B_zy):
    """Compute the trace of the evolution operator for different trajectories

    Parameters
    ----------
    index : [type]
        [description]
    l : [type]
        [description]
    angle : [type]
        [description]
    theta_alpha: float
        Rashba SOI coeffecient
    theta_beta3: float
        Cubic Dresselhaus coeffecient
    theta_beta1: float
        Linear Dresselhaus coeffecient
    N_orbit : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    """
    T = np.empty(N_orbit)


    for n in prange(N_orbit//1000):
        for i in range(1000):
            traj_id = n*1000 + i
            if traj_id >= N_orbit:
                break
            begin, end = index[traj_id]
            T_a = find_trace_Zeeman(l[begin:end], c_phi[begin:end], c_3phi[begin:end], s_phi[begin:end], s_3phi[begin:end],
                                   theta_alpha, theta_beta3, theta_beta1, B_zx, B_zy)
            T[traj_id] = T_a


    return T
