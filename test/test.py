import stab_rs.linear as linear # type: ignore
import matplotlib.pyplot as plt
import numpy as np

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

geom = linear.Geometry(tot_len, diam, ix, iy, xcg, 62)

sim = linear.Simulation(
    geom, "for006_coeffs.hdf5", 100, 890, 7, 3, 30000, 298, 1.225)

sim.run()

sim.write_file()

fig, ax = plt.subplots(2, 1)
ax[0].plot(sim.vec_data.sp_vec, np.array(
    sim.vec_data.alpha_vec)*180.0/np.pi)
ax[0].set_title('Alpha')
ax[0].set_xlabel('Downrange [cal]')
ax[0].set_ylabel('Alpha [deg]')
ax[1].plot(sim.vec_data.sp_vec, np.array(
    sim.vec_data.beta_vec)*180.0/np.pi)
ax[1].set_title('Beta')
ax[1].set_ylabel('Beta [deg]')
ax[1].set_xlabel('Downrange [cal]')
ax[0].grid(True)
ax[1].grid(True)
plt.tight_layout()


fig, ax = plt.subplots()
ax.plot(sim.vec_data.sp_vec, np.array(
    sim.vec_data.alpha_tot_vec)*180.0/np.pi)
ax.set_title('Alpha Total vs downrange distance')
ax.set_xlabel('Downrange [cal]')
ax.set_ylabel('Alpha Total [deg]')
ax.grid(True)

fig, ax = plt.subplots()
line, = ax.plot(np.array(sim.vec_data.beta_vec)*180.0/np.pi,
                np.array(sim.vec_data.alpha_vec)*180.0/np.pi,
                label='_nolegend_')
z = np.linspace(min(np.array(sim.vec_data.alpha_tot_vec)*180.0/np.pi), max(
    np.array(sim.vec_data.alpha_tot_vec)*180.0/np.pi),
    len(sim.vec_data.alpha_vec))
scat = ax.scatter(np.flip(np.array(sim.vec_data.beta_vec)*180.0/np.pi),
                  np.flip(np.array(sim.vec_data.alpha_vec)*180.0/np.pi),
                  c=z, s=6,
                  cmap='viridis')
ax.set_title('Tip trajectory')
ax.set_xlabel('Yaw [deg]')
ax.set_ylabel('Pitch [deg]')
ax.axis('equal')
ax.grid(True)
# Highlight start and end points
ax.scatter(sim.vec_data.beta_vec[0]*180.0/np.pi,
           sim.vec_data.alpha_vec[0]*180.0/np.pi, s=25, c='green',
           edgecolor='green', label='Start')
ax.scatter(sim.vec_data.beta_vec[-1]*180.0/np.pi,
           sim.vec_data.alpha_vec[-1]*180.0/np.pi, s=25, c='red',
           edgecolor='red', label='End')

fig, ax = plt.subplots()
ax.plot(sim.vec_data.sp_vec, sim.vec_data.beta_r_vec)
ax.set_title('Yaw of repose vs downrange distance')
ax.set_xlabel('Downrange [cal]')
ax.set_ylabel('Yaw of repose [deg]')
ax.grid(True)


fig, ax = plt.subplots()
ax.plot(sim.vec_data.sp_vec, sim.vec_data.sd_vec, label='Sd')
ax.plot(sim.vec_data.sp_vec, sim.vec_data.sg_vec, label='Sg')
ax.set_title('Stability coefficients')
ax.set_xlabel('sV')
ax.set_ylabel('Sd,Sg')
ax.grid(True)
ax.legend()

fig, ax = plt.subplots()
ax.plot(sim.vec_data.sp_vec, sim.vec_data.cd, label='Cd')
ax.set_title('Drag')
ax.set_xlabel('sV')
ax.set_ylabel('Cd')
ax.grid(True)
ax.legend()


fig, ax = plt.subplots()
ax.plot(sim.vec_data.sp_vec, sim.vec_data.mach_vec, label='Cd')
ax.set_title('Mach')
ax.set_xlabel('sV')
ax.set_ylabel('Cd')
ax.grid(True)
ax.legend()


sdvec = np.linspace(0.00001, 1.9999, 1000)
sgvec = 1.0/(sdvec*(2-sdvec))

fig, ax = plt.subplots()
ax.plot(sdvec, sgvec)
z = np.linspace(min(sim.vec_data.sg_vec), max(
    sim.vec_data.sg_vec), len(sim.vec_data.sg_vec))
ax.scatter(sim.vec_data.sd_vec, sim.vec_data.sg_vec, c=z, s=20,
           cmap='viridis')

ax.set_xlim([0.0, 2.0])
ax.set_ylim([0.0, 6.0])
ax.grid(True)
ax.set_xlabel('Sd, dynamic stability')
ax.set_title('Stability Factors')
ax.set_ylabel('Sg, gyro stability')
plt.tight_layout()
plt.show()
