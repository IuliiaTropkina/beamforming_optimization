import numpy as np
import matplotlib.pyplot as plt
x = np.zeros([500, 100])

x[200:300, :] = 1
x[300:400, :] = 2
x += np.random.randn(500, 100)
plt.imshow(x.T)
plt.colorbar()

plt.show()