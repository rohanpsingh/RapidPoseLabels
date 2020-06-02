import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.offsetbox import AnchoredText

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

x = range(8,13)
obj1 = [0.0044867684620639855, 0.004681421636515164, 0.006481791178890382, 0.00535828022399982, 0.0043950500706790565]
obj2 = [0.005016178940944574, 0.00749706780627055, 0.007938456035612777, 0.009430220634205035, 0.006047969165020261]
obj3 = [0.0036087381547738303, 0.005608312487404514, 0.003821127163942744, 0.0031501784995833088, 0.0032669949759357403]

f, ax = plt.subplots(1, 2)

anchored_text = AnchoredText("scenes=6", loc=3)
ax[0].add_artist(anchored_text)
ax[0].plot(x, obj1, 'r-*', label='OBJ1')
ax[0].plot(x, obj2, 'b-*', label='OBJ2')
ax[0].plot(x, obj3, 'g-*', label='OBJ3')
ax[0].set_title('Mean error vs Num. of Total Keypoints', fontsize=18)
ax[0].set(ylabel='Mean error (in meters)', xlabel='Total number of defined keypoints')
ax[0].legend(loc="upper left")
ax[0].grid(True)
ax[0].set_ylim(0.002, 0.015)


x = range(3, 9)
obj1 = [0.005584287740979281, 0.004610972510562129, 0.0038908188134693785, 0.004681421636515164, 0.004768553182583737, 0.00535028530681535]
obj2 = [0.007542314567197702, 0.006046332021161252, 0.006316062567719461, 0.005310572205526536, 0.005953525774769443, 0.00612713523042035]
obj3 = [0.008133007244266195, 0.0067365581413224886, 0.006597133963384009, 0.005608312487404514, 0.005320468594074691, 0.004925485970753426]


anchored_text = AnchoredText("keypoints=9", loc=3)
ax[1].add_artist(anchored_text)
ax[1].plot(x, obj1, 'r-*', label='OBJ1')
ax[1].plot(x, obj2, 'b-*', label='OBJ2')
ax[1].plot(x, obj3, 'g-*', label='OBJ3')
ax[1].set_title('Mean error vs Num. of Scenes', fontsize=18)
ax[1].set(xlabel='Total number of scenes')
ax[1].legend(loc="upper left")
ax[1].grid(True)
ax[1].set_ylim(0.002, 0.015)

plt.show()
