import stab_rs  # Module for stability-related calculations
import numpy as np  # For numerical and array operations
from geometry import Geometry  # Geometry class to handle geometrical configurations


class Position:
    """
    Represents the position and orientation parameters.

    Attributes:
        altitude (float): Altitude value (e.g., in meters).
        latitude (float): Latitude value (e.g., in degrees).
        azimut (float): Azimuth value (e.g., in degrees or radians).
        quad_elev (float): Quadrant elevation angle (e.g., in degrees or radians).
        delta_azi (float): Change in azimuth (e.g., in degrees or radians).
    """
    altitude: float
    latitude: float
    azimut: float
    quad_elev: float
    delta_azi: float

    def __new__(cls, altitude: float, latitude: float,
                azimut: float, quad_elev: float, delta_azi: float):
        """
        Creates a new instance of Position.

        Args:
            altitude (float): The altitude value.
            latitude (float): The latitude value.
            azimut (float): The azimuth value.
            quad_elev (float): The quadrant elevation angle.
            delta_azi (float): The delta azimuth.

        Returns:
            Position: A new instance of the Position class.
        """
        ...


class SimData:
    """
    Holds the simulation data vectors.

    Attributes:
        x1, x2, x3 (np.ndarray): Position coordinates.
        u1, u2, u3 (np.ndarray): Velocities.
        udot1, udot2, udot3 (np.ndarray): Accelerations.
        roll (np.ndarray): Roll angle.
        roll_rate (np.ndarray): Roll rate.
        time (np.ndarray): Time points of the simulation.
    """
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


class Simulation:
    """
    Manages and executes a simulation run.

    Attributes:
        geometry (Geometry): The geometrical configuration of the simulation.
        position (Position): The initial position and orientation parameters.
        range (float): The range for the simulation (e.g., in meters).
        twist_rate (float): The rate of twisting motion (angular velocity).
        delta_yaw (float): Change in yaw angle (e.g., in degrees or radians).
        roll_rate (float): Roll rate (angular velocity).
        init_vel (float): Initial velocity for the simulation (e.g., in m/s).
        iterations (int): Number of iterations for the simulation run.
        vec_data (SimData): Data storage for simulation results.
        coriolis (bool): Indicates whether Coriolis effect is considered.
    """
    geometry: Geometry
    position: Position
    range: float
    twist_rate: float
    delta_yaw: float
    roll_rate: float
    init_vel: float
    iterations: int
    vec_data: SimData
    coriolis: bool

    def __new__(cls, geom: Geometry, aero_path: str, position: Position,
                range: float, vel: float, twist: float, delta_yaw: float, coriolis: None):
        """
        Initializes a new simulation instance.

        Args:
            geom (Geometry): The geometrical configuration.
            aero_path (str): Path to the aerodynamic data file.
            position (Position): Initial position and orientation.
            range (float): The range of the simulation.
            vel (float): Initial velocity.
            twist (float): Twist rate.
            delta_yaw (float): Change in yaw angle.
            coriolis (bool): Whether to include Coriolis effect.

        Returns:
            Simulation: A new instance of the Simulation class.
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
        Updates the geometry for the simulation.

        Args:
            geom (Geometry): New geometrical configuration.
        """
        ...

    def run(self) -> None:
        """
        Executes the simulation using the current parameters and data.
        """
        ...

    def write_results(self) -> None:
        """
        Writes the results of the simulation to an output file "sim_stanag.hdf5".
        """
        ...
