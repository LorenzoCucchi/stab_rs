# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import stab_rs
import numpy as np


class Position:
    altitude: float
    latitude: float
    azimut: float
    quad_elev: float
    delta_azi: float
    def __new__(cls, altitude: float, latitude: float,
                azimut: float, quad_elev: float, delta_azi: float): ...


class SimData:
    x1: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    u1: np.ndarray
    u2: np.ndarray
    u3: np.ndarray
    udot1: np.ndarray
    udot2: np.ndarray
    udot3: np.ndarray
    roll: np.ndarray
    roll_rate: np.ndarray
    time: np.ndarray


class Geometry:
    length: float
    diameter: float
    area: float
    in_x: float
    in_y: float
    xcg: float
    mass: float
    mass_grain: float

    def __init__(length: float, diameter: float, in_x: float, in_y: float,
                 xcg: float, mass_grain: float) -> Geometry: ...

    def print(self): ...


class Simulation:
    geometry: Geometry
    position: Position
    range: float
    twist_rate: float
    delta_yaw: float
    roll_rate: float
    init_vel: float
    iterations: int
    vec_data: SimData

    def __new__(cls, geom: Geometry, aero_path: str, position: Position,
                range: float, vel: float, twist: float, delta_yaw: float): ...

    def change_aerodata(self, path: str) -> None:
        ...

    def change_geometry(self, geom: Geometry) -> None:
        ...

    def run(self) -> None:
        ...

    def write_results(self) -> None:
        ...
