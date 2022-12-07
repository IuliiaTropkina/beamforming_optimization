import matplotlib.pyplot as plt
import numpy as np

#M = np.zeros([500,500,3], dtype=np.uint8)
M = np.full([500,500,3],255, dtype=np.uint8)

M[10:10+10,255:250+10,:] = 0
plt.imshow(M,origin='lower')
plt.show()