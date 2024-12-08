import numpy as np  # For numerical and array operations
from geometry import Geometry  # For geometrical configurations


class NamedVec:
    """
    Represents a named vector containing a name and associated numerical data.

    Attributes:
        name (str): The name of the vector.
        data (np.ndarray): The numerical data of the vector.
    """
    name: str
    data: np.ndarray


class Aero:
    """
    Handles aerodynamic data and operations.

    Attributes:
        path (str): Path to the aerodynamic data file.
        alpha (np.ndarray): Array of angle-of-attack values.
        mach (np.ndarray): Array of Mach number values.
        cd_fit (np.ndarray): Fitted drag coefficient data.
        cna_fit (np.ndarray): Fitted normal force coefficient data.
        cma_fit (np.ndarray): Fitted pitching moment coefficient data.
        cnpa_fit (np.ndarray): Fitted normal force moment derivative data.
        cd, cna, cma, cnpa (np.ndarray): Actual aerodynamic coefficients.
    """

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

    def __init__(self, path: str) -> "Aero":
        """
        Initializes the Aero class with aerodynamic data from the specified path.

        Args:
            path (str): Path to the aerodynamic data file.
        """
        ...

    def interp1(self, mach: float):
        """
        Interpolates aerodynamic data for a given Mach number.

        Args:
            mach (float): Mach number for interpolation.
        """
        ...

    def update(self, mach: float):
        """
        Updates the aerodynamic coefficients based on the given Mach number.

        Args:
            mach (float): Current Mach number.
        """
        ...

    def get_cd(self, alpha: float) -> float:
        """
        Retrieves the drag coefficient for a given angle of attack.

        Args:
            alpha (float): Angle of attack.

        Returns:
            float: Drag coefficient.
        """
        ...

    def get_cna(self, alpha: float) -> float:
        """
        Retrieves the normal force coefficient for a given angle of attack.

        Args:
            alpha (float): Angle of attack.

        Returns:
            float: Normal force coefficient.
        """
        ...

    def get_cma(self, alpha: float) -> float:
        """
        Retrieves the pitching moment coefficient for a given angle of attack.

        Args:
            alpha (float): Angle of attack.

        Returns:
            float: Pitching moment coefficient.
        """
        ...

    def get_cnpa(self, alpha: float) -> float:
        """
        Retrieves the normal force moment derivative coefficient for a given angle of attack.

        Args:
            alpha (float): Angle of attack.

        Returns:
            float: Normal force moment derivative.
        """
        ...


class SimData:
    """
    Stores various simulation data vectors using NamedVec.

    Attributes:
        sp_vec, vel_vec, mach_vec, etc. (NamedVec): Various named vectors for simulation data.
    """
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

    @staticmethod
    def new() -> "SimData":
        """
        Creates a new instance of SimData with default values.

        Returns:
            SimData: A new SimData instance.
        """
        ...


class Simulation:
    """
    Manages the setup, execution, and results of a simulation.

    Attributes:
        geometry (Geometry): Geometry configuration for the simulation.
        range_meters (float): Effective range in meters.
        range_max (float): Maximum range.
        twist_rate (float): Twist rate of the object.
        delta_yaw (float): Change in yaw angle.
        roll_rate (float): Roll rate of the object.
        temperature (float): Ambient temperature in degrees Celsius.
        density (float): Air density in kg/m^3.
        sound (float): Speed of sound at the given temperature.
        init_vel (float): Initial velocity.
        init_mach (float): Initial Mach number.
        iterations (int): Number of simulation iterations.
        vec_data (SimData): Simulation data storage.
        aero_data (Aero): Aerodynamic data storage and processing.
    """
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

    def __new__(cls, geom: Geometry, aero_path: str, range: float, vel: float,
                twist: float, delta_yaw: float, iter: int, temp: float, density: float) -> "Simulation":
        """
        Initializes a new simulation instance.

        Args:
            geom (Geometry): Geometrical configuration.
            aero_path (str): Path to aerodynamic data.
            range (float): Simulation range.
            vel (float): Initial velocity.
            twist (float): Twist rate.
            delta_yaw (float): Yaw angle change.
            iter (int): Number of iterations.
            temp (float): Ambient temperature.
            density (float): Air density.

        Returns:
            Simulation: A new Simulation instance.
        """
        ...

    @staticmethod
    def sound_speed(temp: float) -> float:
        """
        Calculates the speed of sound based on temperature.

        Args:
            temp (float): Temperature in degrees Celsius.

        Returns:
            float: Speed of sound in m/s.
        """
        ...

    def set_conditions(self, range: float, vel: float, twist: float, delta_yaw: float,
                       iter: int, temp: float, density: float):
        """
        Sets the initial conditions for the simulation.

        Args:
            range (float): Simulation range.
            vel (float): Initial velocity.
            twist (float): Twist rate.
            delta_yaw (float): Yaw angle change.
            iter (int): Number of iterations.
            temp (float): Ambient temperature.
            density (float): Air density.
        """
        ...

    def init_vectors(self):
        """
        Initializes the vectors for storing simulation data.
        """
        ...

    def trajectory(self, p: float, m: float, t: float, g: float, h: float,
                   xi0_re: float, xi0_im: float, xi0_pr_re: float, xi0_pr_im: float,
                   s: float, d: float, i: int):
        """
        Computes the trajectory based on initial conditions and parameters.

        Args:
            p, m, t, g, h (float): Physical parameters.
            xi0_re, xi0_im (float): Initial position (real and imaginary parts).
            xi0_pr_re, xi0_pr_im (float): Initial velocity (real and imaginary parts).
            s, d (float): Drag and stability parameters.
            i (int): Iteration index.
        """
        ...

    def run(self):
        """
        Executes the simulation.
        """
        ...

    def write_file(self):
        """
        Writes the simulation results to a file.
        """
        ...
