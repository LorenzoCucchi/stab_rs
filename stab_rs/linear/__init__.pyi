import numpy as np
from geometry import Geometry

class NamedVec:
    name: str
    data: np.ndarray

class Aero:
    path: str
    alpha: np.ndarray
    mach: np.ndarray
    cd_fit: np.ndarray
    cna_fit: np.ndarray
    cma_fit: np.ndarray
    cnpa_fit: np.ndarray
    cd: np.ndarray
    cna: np.ndarray
    cma: np.ndarray
    cnpa: np.ndarray

    def __init__(path: str) -> Aero: ...

    def interp1(self, mach: float): ...

    def update(self, mach: float): ...

    def get_cd(self, alpha: float) -> float: ...

    def get_cna(self, alpha: float) -> float: ...

    def get_cma(self, alpha: float) -> float: ...

    def get_cnpa(self, alpha: float) -> float: ...


class SimData:
    sp_vec: NamedVec
    vel_vec: NamedVec
    mach_vec: NamedVec
    p_vec: NamedVec
    sg_vec: NamedVec
    sd_vec: NamedVec
    sg_lim_vec: NamedVec
    dr_vec: NamedVec
    alpha_vec: NamedVec
    beta_vec: NamedVec
    alpha_tot_vec: NamedVec
    beta_r_vec: NamedVec
    lambda_f_vec: NamedVec
    lambda_s_vec: NamedVec
    fast_freq_vec: NamedVec
    slow_freq_vec: NamedVec
    cd: NamedVec
    cna: NamedVec
    cma: NamedVec
    cmqcmad: NamedVec
    def new() -> SimData: ...


class Simulation:
    geometry: Geometry
    range_meters: float
    range_max: float
    twist_rate: float
    delta_yaw: float
    roll_rate: float
    temperature: float
    density: float
    sound: float
    init_vel: float
    init_mach: float
    iterations: int
    vec_data: SimData
    aero_data: Aero

    def __new__(geom: Geometry, aero_path: str, range: float, vel: float,
                 twist: float, delta_yaw: float, iter: int, temp: float, density: float) -> Simulation: ...

    def sound_speed(temp: float) -> float: ...

    def set_conditions(self, range: float, vel: float, twist: float, delta_yaw: float,
                       iter: int, temp: float, density: float): ...

    def init_vectors(self): ...

    def trajectory(self, p: float,
                   m: float,
                   t: float,
                   g: float,
                   h: float,
                   xi0_re: float,
                   xi0_im: float,
                   xi0_pr_re: float,
                   xi0_pr_im: float,
                   s: float,
                   d: float,
                   i: int): ...

    def run(self): ...

    def write_file(self): ...
