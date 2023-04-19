from numpy import array
import pandas as pd
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import signal
warnings.filterwarnings("ignore")
# load the dataset


dataset1 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Position/OpenHand/open_hand_1.csv')
dataset2 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Position/OpenHand/open_hand_2.csv')
dataset3 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Position/OpenHand/open_hand_3.csv')
dataset4 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Position/OpenHand/open_hand_4.csv')
dataset5 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Position/OpenHand/open_hand_5.csv')
data1=array(dataset1)
data2=array(dataset2)
data3=array(dataset3)
data4=array(dataset4)
data5=array(dataset5)
X1 = data1[:,:]
X2 = data2[:,:]
X3 = data3[:,:]
X4 = data4[:,:]
X5 = data5[:,:]
fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1)

ax1.set_ylabel('O1')
# ax3.set_ylim([-2, 6])
# ax3.axhline(y=0.5, color='blue')
# ax3.axhline(y=1, color='red')
ax1.plot(X1[:, 0], label='Labels 0', color='green')
ax1.plot(X1[:, 1], label='Column 2', color='blue')

ax2.set_ylabel('O2')
ax2.plot(X2[:, 0], label='Labels 0', color='green')
ax2.plot(X2[:, 1], label='Column 2', color='blue')

ax3.set_ylabel('O3')
ax3.plot(X3[:, 0], label='Labels 0', color='green')
ax3.plot(X3[:, 1], label='Column 2', color='blue')

ax4.set_ylabel('O4')
ax4.plot(X4[:, 0], label='Labels 0', color='green')
ax4.plot(X4[:, 1], label='Column 2', color='blue')

ax5.set_ylabel('O4')
ax5.plot(X5[:, 0], label='Labels 0', color='green')
ax5.plot(X5[:, 1], label='Column 2', color='blue')

plt.grid()
plt.show()
