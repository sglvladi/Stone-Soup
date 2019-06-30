import numpy as np
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
points = np.random.random((2, 1000))

tree = cKDTree(points.T)

# plt.figure()
# plt.plot(points[0,:], points[1,:], 'r.')

# plt.pause(0.0001)

