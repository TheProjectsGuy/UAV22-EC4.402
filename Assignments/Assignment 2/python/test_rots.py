# Testing rotations
"""
"""

# %%
import sympy as sp
import numpy as np

# %%
# Roll
def rot_x(angle, degrees = False):
    """
    Generates a Rotation matrice when the rotation is about X axis
    A general rotation about X (Roll) is given by
                | 1    0    0 |
    RotX(T) =   | 0   cT  -sT |
                | 0   sT   cT |
    Where T is rotation angle in radians
    Parameters:
    - angle: Symbol or float
        The angle of rotation
    - degrees: bool     default: False
        If 'True', then angle parameter is assumed to be in degrees
        else it is by default assumed to be in radians
    
    Returns:
    - rot_mat: sp.Matrix        shape: (3, 3)
        The 3x3 rotation matrix for Roll by angle_rad
    """
    angle_rad = angle if not degrees else sp.rad(angle)
    rot_mat = sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(angle_rad), -sp.sin(angle_rad)],
        [0, sp.sin(angle_rad), sp.cos(angle_rad)],
    ])
    return rot_mat


# Pitch
def rot_y(angle, degrees = False):
    """
    Generates a Rotation matrice when the rotation is about Y axis
    A general rotation about Y (Pitch) is given by
                |  cT   0   sT |
    RotY(T) =   |   0   1    0 |
                | -sT   0   cT |
    Where T is rotation angle in radians
    Parameters:
    - angle: Symbol or float
        The angle of rotation
    - degrees: bool     default: False
        If 'True', then angle parameter is assumed to be in degrees
        else it is by default assumed to be in radians
    
    Returns:
    - rot_mat: sp.Matrix        shape: (3, 3)
        The 3x3 rotation matrix for Pitch by angle_rad
    """
    angle_rad = angle if not degrees else sp.rad(angle)
    rot_mat = sp.Matrix([
        [sp.cos(angle_rad), 0, sp.sin(angle_rad)],
        [0, 1, 0],
        [-sp.sin(angle_rad), 0, sp.cos(angle_rad)],
    ])
    return rot_mat


# Yaw
def rot_z(angle, degrees = False):
    """
    Generates a Rotation matrice when the rotation is about Z axis
    A general rotation about Z (Yaw) is given by
                | cT   -sT   0 |
    RotZ(T) =   | sT    cT   0 |
                |  0     0   1 |
    Where T is rotation angle in radians
    Parameters:
    - angle: Symbol or float
        The angle of rotation
    - degrees: bool     default: False
        If 'True', then angle parameter is assumed to be in degrees
        else it is by default assumed to be in radians
    
    Returns:
    - rot_mat: sp.Matrix        shape: (3, 3)
        The 3x3 rotation matrix for Yaw by angle_rad
    """
    angle_rad = angle if not degrees else sp.rad(angle)
    rot_mat = sp.Matrix([
        [sp.cos(angle_rad), -sp.sin(angle_rad), 0],
        [sp.sin(angle_rad), sp.cos(angle_rad), 0],
        [0, 0, 1],
    ])
    return rot_mat

# %%
y,p,r = sp.symbols(r"\psi, \theta, \phi")
# Body (local) ZYX === Fixed inertial (global) XYZ angles
rot_m = rot_z(y) @ rot_y(p) @ rot_x(r)
# Testing
a,b,g = sp.symbols(r"\alpha, \beta, \gamma")
rot_m1 = rot_z(a) @ rot_y(b) @ rot_x(g)

# %% Thrust vector
T = sp.Matrix([[0], [0], [sp.symbols("T")]])   # In UAV frame
T_g = rot_m @ T     # In inertial frame

# %%
m = sp.symbols("m")
g_vect = sp.Matrix([[0], [0], [sp.symbols("g")]])
# 
accs = g_vect + (T_g/m)


# %%


# %%
