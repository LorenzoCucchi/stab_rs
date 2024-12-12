import stab_rs.stanag as stanag  # type: ignore
import stab_rs.geometry as geometry
import stab_rs.wind as wind
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

geomS = geometry.Geometry(tot_len, diam, ix, iy, xcg, 62)

windS = wind.Wind(10, np.pi/2.0, 0, 2, 0.0001, wind.Turbulence.CONST)

zero_seed = bytes([0] * 32)

windS.fixed_seed(zero_seed, zero_seed, zero_seed)

posS = stanag.Position(1.0, 0.0, 0.0, 0.098*np.pi/180, 0)

posS.azimut = 0.0
posS.latitude = 0.0

simS = stanag.Simulation(geomS, "coeffs.json", posS, windS, 0.0001,
                         200.0, 900.0, 7.0, 4.0, False)

simS.run()

simS.write_results()
