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
dataset0 = pd.read_csv('k2.csv')
data0=array(dataset0)

X0 = data0[:,:]

fig3,(ax3) = plt.subplots(1, 1)

ax3.set_ylabel('Emg Signal')
ax3.set_ylim([-2, 6])
# ax3.axhline(y=0.5, color='blue')
# ax3.axhline(y=1, color='red')
ax3.plot(X0[:, 0], label='Labels 0', color='green')
ax3.plot(X0[:, 1], label='Column 2', color='black')

plt.grid()
plt.show()
