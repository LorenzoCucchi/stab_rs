import numpy as np
import matplotlib.pyplot as plt
import secrets
# Function to generate Dryden turbulence


def dryden_turbulence(time, V, Lu, Lv, Lw, sigma_u, sigma_v, sigma_w, dt):
    """
    Simulate Dryden turbulence using discrete-time difference equations.

    Parameters:
    - time: Total simulation time (s)
    - V: Aircraft speed (m/s)
    - Lu, Lv, Lw: Scale lengths (m) for longitudinal, lateral, vertical components
    - sigma_u, sigma_v, sigma_w: RMS intensities (m/s) for the turbulence components
    - dt: Time step for simulation (s)

    Returns:
    - t: Time array
    - ug, vg, wg: Wind turbulence velocities (arrays)
    """
    # Time array
    t = np.arange(0, time, dt)
    n_steps = len(t)

    # Initialize turbulence velocities
    ug = np.zeros(n_steps)
    vg = np.zeros(n_steps)
    wg = np.zeros(n_steps)

    random_seed = secrets.randbelow(2**32)  # Generate a secure random seed
    np.random.seed(1)  # Set the seed for reproducibility

    # Generate white noise inputs
    eta_u = np.random.normal(0, 1, n_steps)
    eta_v = np.random.normal(0, 1, n_steps)
    eta_w = np.random.normal(0, 1, n_steps)

    V = V*3.28084
    # Longitudinal turbulence coefficients
    a1_u = -V / Lu
    b1_u = sigma_u * np.sqrt(2 * V / Lu)

    # Lateral turbulence coefficients
    a1_v = -V / Lv
    b1_v = sigma_v * np.sqrt(2 * V / Lv)

    # Vertical turbulence coefficients
    a1_w = -V / Lw
    b1_w = sigma_w * np.sqrt(2 * V / Lw)

    # Iterate over time to compute turbulence velocities
    for k in range(1, n_steps):
        # Longitudinal
        ug[k] = ug[k-1] + a1_u * ug[k-1] * dt + b1_u * np.sqrt(dt) * eta_u[k-1]
        # Lateral
        vg[k] = vg[k-1] + a1_v * vg[k-1] * dt + b1_v * np.sqrt(dt) * eta_v[k-1]
        # Vertical
        wg[k] = wg[k-1] + a1_w * wg[k-1] * dt + b1_w * np.sqrt(dt) * eta_w[k-1]

    return t, ug, vg, wg


def scale_lengths(altitude: float, MIL_HDBK: bool):
    """
    Calculate low altitude scale lengths

    Args:
      altitude (float): altitude for calculations
      MIL_HDBK (bool): if true use MIL-HDBK-1797 if False use MIL-F-8785C
    Returns:
        (Lw, Lu, Lv) (float, float, float): Scale lengths
    """
    altitude = altitude*3.28084
    Lu = altitude/(0.177+0.000823*altitude)**1.2

    if MIL_HDBK:
        Lw = altitude/2.0
        Lv = Lu/2.0
    else:
        Lw = altitude
        Lv = Lu

    return Lw, Lu, Lv


def turb_intensity(wind, altitude):
    """
    Calculate turbulence intensity given in input wind speed and altitude

    Args:
        wind (float): wind speed in [m/s]
        altitude (float): altitude in [m]
    Returns:
        (sigma_w,sigma_u,sigma_v) (float, float, float): turbulence intensities

    """
    sigma_w = 0.1*wind*3.28084
    sigma_u = 1/(0.177+0.000823*altitude*3.28084)**0.4 * sigma_w
    sigma_v = sigma_u

    return sigma_w, sigma_u, sigma_v


sigma_w, sigma_u, sigma_v = turb_intensity(4, 5)
# Example parameters
time = 0.5  # Total simulation time in seconds
V = 500  # Aircraft speed in m/s
Lw, Lu, Lv = scale_lengths(5, False)
print(Lw, Lu, Lv)
dt = 0.00001  # Time step in seconds

# Generate turbulence
t, ug, vg, wg = dryden_turbulence(
    time, V, Lu, Lv, Lw, sigma_u, sigma_v, sigma_w, dt)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, ug/3.28084, label='Longitudinal (u_g)')
plt.plot(t, vg/3.28084, label='Lateral (v_g)')
plt.plot(t, wg/3.28084, label='Vertical (w_g)')
plt.title("Dryden Wind Turbulence")
plt.xlabel("Time (s)")
plt.ylabel("Wind Velocity Disturbance (m/s)")
plt.legend()
plt.grid()
plt.show()
