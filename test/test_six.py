import stab_rs.sixdof as sixdof  # type: ignore
import stab_rs.geometry as geometry
import matplotlib.pyplot as plt
import numpy as np


def calc_state(vel: float, rot: float, pitch: float, yaw: float, delta: float, phase: float):
    vx = vel*np.cos(delta)
    vy = vel*np.sin(delta)*np.sin(phase)
    vz = vel*np.sin(delta)*np.cos(phase)

    p = rot*np.cos(delta)
    q = rot*np.sin(delta)*np.sin(phase)
    r = rot*np.sin(delta)*np.cos(phase)

    delta_pitch = np.atan(vz/vx)
    delta_yaw = np.atan(vy/vx)

    pitch = pitch + delta_pitch
    yaw = yaw + delta_yaw

    return vx, -vy, vz, pitch, yaw, -q, r


mass = 0.00404
xcg = 0.014279
ix = 0.1416*10e-8
iy = 1.138*10e-8
diam = 0.00569
lnose = 0.01269
lcentr = 0.007966
laft = 0.002788
daft = 0.004958

tot_len = lnose+lcentr+laft

geomS = geometry.Geometry(tot_len, diam, ix, iy, xcg, 62)


simS = sixdof.Simulation(geomS, "for006_coeffs.hdf5", 0.00001, 0.29, 8.0)

vx, vy, vz, pitch, yaw, q, r = calc_state(
    vel=800, rot=24730, pitch=0.11*np.pi/180.0, yaw=0.0, delta=0.1*np.pi/180.0, phase=225.0*np.pi/180)

print("Vx: ",vx," Vy: ",vy," Vz: ", vz)
print("pitch: ",pitch," yaw: ",yaw)
print("q: ",q," r: ",r)
simS.init_state(vx, vy, vz, pitch, 0.0, yaw, q, r)
simS.run()

simS.write_results()


alpha_tot = np.sqrt(np.array(simS.vec_data.alpha) **
                    2+np.array(simS.vec_data.beta)**2)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot()
ax.plot(simS.vec_data.x_n, alpha_tot*180.0/np.pi)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(np.array(simS.vec_data.x_n), np.array(simS.vec_data.beta)*180.0/np.pi,
        np.array(simS.vec_data.alpha)*180.0/np.pi, label='Trajectory', color='blue')
# ax.scatter(simS.vec_data.x_n[0], simS.vec_data.beta[0], simS.vec_data.alpha[0], color='green', label='Start Point')  # Mark the start point
# ax.scatter(simS.vec_data.x_n[-1], simS.vec_data.beta[-1], simS.vec_data.alpha[-1], color='red', label='End Point')   # Mark the end point

# Set labels
ax.set_xlabel('Range [m]')
ax.set_ylabel('Beta [deg°]')
ax.set_zlabel('Alpha [deg°]')
ax.set_title('Alpha/Beta ')

# Add legend
ax.legend()


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot()
ax.plot(np.array(simS.vec_data.x_n), -np.array(simS.vec_data.x_d))
ax.set_xlabel("X North [m]")
ax.set_ylabel("X Down [m]")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot()
ax.plot(np.array(simS.vec_data.x_n), np.array(simS.vec_data.x_e))
ax.set_xlabel("X North [m]")
ax.set_ylabel("X East [m]")
# Show the plot
plt.show()
