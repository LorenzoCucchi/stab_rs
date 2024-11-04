import numpy as np


class Geometry:
    length: float
    diameter: float
    area: float
    in_x: float
    in_y: float
    xcg: float
    mass: float
    mass_grain: float

    def new(length: float, diameter: float, in_x: float, in_y: float,
            xcg: float, mass_grain: float) -> Geometry: ...

    def print(self): ...


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

    def new(path: str) -> Aero: ...

    def interp1(self, mach: float): ...

    def update(self, mach: float): ...

    def get_cd(self, alpha: float) -> float: ...

    def get_cna(self, alpha: float) -> float: ...

    def get_cma(self, alpha: float) -> float: ...

    def get_cnpa(self, alpha: float) -> float: ...


class SimData:
    sp_vec: np.ndarray
    vel_vec: np.ndarray
    mach_vec: np.ndarray
    p_vec: np.ndarray
    sg_vec: np.ndarray
    sd_vec: np.ndarray
    sg_lim_vec: np.ndarray
    dr_vec: np.ndarray
    alpha_vec: np.ndarray
    beta_vec: np.ndarray
    alpha_tot_vec: np.ndarray
    beta_r_vec: np.ndarray
    lambda_f_vec: np.ndarray
    lambda_s_vec: np.ndarray
    fast_freq_vec: np.ndarray
    slow_freq_vec: np.ndarray
    cd: np.ndarray
    cna: np.ndarray
    cma: np.ndarray
    cmqcmad: np.ndarray
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

    def new(geom: Geometry, aero_path: str, range: float, vel: float,
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
