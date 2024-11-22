import stab_rs
import numpy as np
from geometry import Geometry


class SimData:
    x_n: np.ndarray
    x_e: np.ndarray
    x_d: np.ndarray
    v_n: np.ndarray
    v_e: np.ndarray
    v_d: np.ndarray
    v_bx: np.ndarray
    v_by: np.ndarray
    v_bz: np.ndarray
    quat_x: np.ndarray
    quat_y: np.ndarray
    quat_z: np.ndarray
    quat_w: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    p: np.ndarray
    q: np.ndarray
    r: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray


class Simulation:
    dt: float
    ft: float
    geometry: Geometry
    twist_rate: float
    vec_data: SimData

    def __new__(cls, geom: Geometry, aero_path: str, dt: float,
                ft: float, twist: float): ...

    def change_aerodata(self, path: str) -> None:
        ...

    def change_geometry(self, geom: Geometry) -> None:
        ...

    def init_state(self, vx: float, vy: float, vz: float, pitch: float,
                   roll: float, yaw: float, q: float, r: float): ...

    def run(self) -> None:
        ...

    def write_results(self) -> None:
        ...
