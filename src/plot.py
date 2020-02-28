import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

x = range(6,13)
y = [np.random.uniform(low=0.005, high=0.01, size=len(x)) for i in range(3)]

f, ax = plt.subplots(1, 2)

anchored_text = AnchoredText("scenes=8", loc=3)
ax[0].add_artist(anchored_text)
ax[0].plot(x, y[0], 'r-*', label='OBJ1')
ax[0].plot(x, y[1], 'b-*', label='OBJ2')
ax[0].plot(x, y[2], 'g-*', label='OBJ3')
ax[0].set_title('Mean error vs Num. of Total Keypoints')
ax[0].set(ylabel='Mean error (in meters)', xlabel='Total number of defined keypoints')
ax[0].legend(loc="upper left")
ax[0].grid(True)
ax[0].set_ylim(0.002, 0.015)



anchored_text = AnchoredText("keypoints=9", loc=3)
ax[1].add_artist(anchored_text)
ax[1].plot(x, y[0], 'r-*', label='OBJ1')
ax[1].plot(x, y[1], 'b-*', label='OBJ2')
ax[1].plot(x, y[2], 'g-*', label='OBJ3')
ax[1].set_title('Mean error vs Num. of Scenes')
ax[1].set(xlabel='Total number of scenes')
ax[1].legend(loc="upper left")
ax[1].grid(True)
ax[1].set_ylim(0.002, 0.015)

plt.show()
