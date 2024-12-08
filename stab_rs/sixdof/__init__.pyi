import stab_rs
import numpy as np
from geometry import Geometry


class SimData:
    """
    Holds simulation data for state and motion variables.

    Attributes:
        x_n, x_e, x_d (np.ndarray): Position components in the north, east, and down directions.
        v_n, v_e, v_d (np.ndarray): Velocity components in the north, east, and down directions.
        v_bx, v_by, v_bz (np.ndarray): Body-frame velocity components along x, y, and z axes.
        quat_x, quat_y, quat_z, quat_w (np.ndarray): Quaternion components representing orientation.
        phi (np.ndarray): Roll angle.
        theta (np.ndarray): Pitch angle.
        psi (np.ndarray): Yaw angle.
        p, q, r (np.ndarray): Angular velocity components about the body axes (roll, pitch, yaw rates).
        alpha (np.ndarray): Angle of attack.
        beta (np.ndarray): Sideslip angle.
    """
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
    """
    Represents a simulation system that manages setup, execution, and results.

    Attributes:
        dt (float): Time step for the simulation.
        ft (float): Final time or duration of the simulation.
        geometry (Geometry): Geometrical configuration of the system.
        twist_rate (float): The rate of twisting motion (angular velocity).
        vec_data (SimData): Simulation data storage.
    """
    dt: float
    ft: float
    geometry: Geometry
    twist_rate: float
    vec_data: SimData

    def __new__(cls, geom: Geometry, aero_path: str, dt: float,
                ft: float, twist: float):
        """
        Creates a new instance of the Simulation class.

        Args:
            geom (Geometry): Geometrical configuration for the simulation.
            aero_path (str): Path to the aerodynamic data file.
            dt (float): Time step for the simulation.
            ft (float): Final time or duration of the simulation.
            twist (float): Twist rate for the simulation.

        Returns:
            Simulation: A new Simulation instance.
        """
        ...

    def change_aerodata(self, path: str) -> None:
        """
        Updates the aerodynamic data used in the simulation.

        Args:
            path (str): Path to the new aerodynamic data file.
        """
        ...

    def change_geometry(self, geom: Geometry) -> None:
        """
        Updates the geometry configuration for the simulation.

        Args:
            geom (Geometry): New geometrical configuration.
        """
        ...

    def init_state(self, vx: float, vy: float, vz: float, pitch: float,
                   roll: float, yaw: float, q: float, r: float) -> None:
        """
        Initializes the state variables for the simulation.

        Args:
            vx (float): Initial velocity in the x-direction.
            vy (float): Initial velocity in the y-direction.
            vz (float): Initial velocity in the z-direction.
            pitch (float): Initial pitch angle.
            roll (float): Initial roll angle.
            yaw (float): Initial yaw angle.
            q (float): Initial pitch rate.
            r (float): Initial yaw rate.
        """
        ...

    def run(self) -> None:
        """
        Executes the simulation using the current configuration and state variables.
        """
        ...

    def write_results(self) -> None:
        """
        Outputs the results of the simulation to a file "sim_sixdof.hdf5".
        """
        ...
