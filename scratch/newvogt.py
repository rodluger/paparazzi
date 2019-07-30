import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt


doppler = pp.Doppler(ydeg=10)
doppler.generate_data()

doppler.solve(b=doppler.b_true, vT=doppler.vT_true)
plt.plot(doppler.u_true)
plt.plot(doppler.u)
#plt.show()

#doppler.solve(vT=doppler.vT_true, b=doppler.b_true)
doppler.map[1:, :] = doppler.u
doppler.map.show(projection="rect")
