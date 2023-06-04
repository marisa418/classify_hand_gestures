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
from scipy.signal import butter, lfilter

data1 =pd.read_csv('hello/Dataset/all/1f.csv')
X1 = data1.iloc[:1000, :1]
Y1 = data1.iloc[:1000, 1:]

data2=pd.read_csv('hello/Dataset/all/1l.csv')
X2 = data2.iloc[:1000, :1]
Y2 = data2.iloc[:1000, 1:]

data3 = pd.read_csv('hello/Dataset/all/1o.csv')
X3 = data3.iloc[:1000, :1]
Y3 = data3.iloc[:1000, 1:]

data4 = pd.read_csv('hello/Dataset/all/1t.csv')
X4 = data4.iloc[:1000, :1]
Y4 = data4.iloc[:1000, 1:]



import matplotlib.pyplot as plt
fig1,(ax1) = plt.subplots(1, 1)
fig1.suptitle('Fist pose')
ax1.plot(X1, label='Column 2', color='blue')
ax1.plot(Y1, label='Column 2', color='red')
ax1.set_ylim(0,5)
ax1.set_xlabel('Channel 1')
ax1.set_ylabel('Channel 2')

fig2,(ax2) = plt.subplots(1, 1)
fig2.suptitle('Love posture')
ax2.plot(X2, label='Column 2', color='blue')
ax2.plot(Y2, label='Column 2', color='red')
ax2.set_ylim(0,5)
ax2.set_xlabel('Channel 1')
ax2.set_ylabel('Channel 2')

fig3,(ax3) = plt.subplots(1, 1)
fig3.suptitle('Open hand posture')
ax3.plot(X3, label='Column 2', color='blue')
ax3.plot(Y3, label='Column 2', color='red')
ax3.set_ylim(0,5)
ax3.set_xlabel('Channel 1')
ax3.set_ylabel('Channel 2')

fig4,(ax4) = plt.subplots(1, 1)
fig4.suptitle('Clenched fist and raised index finger and middle finger')
ax4.plot(X4, label='Column 2', color='blue')
ax4.plot(Y4, label='Column 2', color='red')
ax4.set_ylim(0,5)
ax4.set_xlabel('Channel 1')
ax4.set_ylabel('Channel 2')


plt.grid()
plt.show()
