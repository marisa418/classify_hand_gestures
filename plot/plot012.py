from numpy import array
import pandas as pd
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
# load the dataset
dataset0 = pd.read_csv('Emg0.csv')
dataset1 = pd.read_csv('Emg1.csv')
dataset2 = pd.read_csv('Emg2.csv')

# split into input (X) and output (y) variables
data0=array(dataset0)
data1=array(dataset1)
data2=array(dataset2)

X0 = data0[:10000,:-1]
X1 = data1[:10000,:-1]
X2 = data2[:10000,:-1]

scaler = StandardScaler()
X0 = scaler.fit_transform(X0)
X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)


T = 1.0 / 9600.0
X0 = X0.reshape(-1)
X1 = X1.reshape(-1)
X2 = X2.reshape(-1)

#fig0
fig0,(ax0) = plt.subplots(1, 1)
fig0.suptitle('Close & Open hand gestures')
ax0.plot(X0)
ax0.set_ylabel('Emg Signal')
ax0.set_ylim([-2, 6])

#fig1
fig1,(ax1) = plt.subplots(1, 1)
fig1.suptitle('Open hand & Thumb touching index finger in a circle')
ax1.plot(X1)
ax1.set_ylabel('Emg Signal')
ax1.set_ylim([-2, 6])

#fig2
fig2,(ax2) = plt.subplots(1, 1)
fig2.suptitle('EMG signal Time & Frequency Domain 50,400')
ax2.plot(X2)
ax2.set_ylabel('Emg Signal')
ax2.set_ylim([-2, 6])

#fig3
fig3,(ax3) = plt.subplots(1, 1)
ax3.plot(X0, label='Labels 0')
ax3.plot(X1, label='Labels 1')
ax3.plot(X2, label='Labels 2')
ax3.set_ylabel('Emg Signal')
ax3.set_ylim([-2, 6])


plt.legend()

plt.show()
