import stab_rs.stanag as stanag  # type: ignore
import stab_rs.geometry as geometry
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

posS = stanag.Position(0,0.0,0,0.0,0)

posS.altitude = 0.0
posS.azimut = 0.0
posS.latitude = 0.0

simS = stanag.Simulation(geomS,"coeffs.json",posS,100.0,800.0,7.0,4.0)

simS.run()

simS.write_results()

