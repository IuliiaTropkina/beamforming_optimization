import numpy as np
import matplotlib.pyplot as plt
c = 3e8
f = 1e9
depth = np.linspace(4, 5, 1000)
radius = (depth - 4) * np.sin(np.pi/4)
tof = np.sqrt(depth**2 + radius**2) / c
nseg = 2 * np.pi * radius * 1000
tau = []
REF = True
for d,r,n, t in zip(depth, radius, nseg, tof):
    tau.extend([t] * int(n))
    REF = not REF
tau = np.array(tau)
power = 1/(tau*c)**4
phase = (2 * np.pi * tau*2 * f) % (2*np.pi)

plt.figure()
plt.hist(phase,weights=power, bins=20)
plt.show()

