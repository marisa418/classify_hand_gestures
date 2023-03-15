from numpy import array
import pandas as pd
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy import signal
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
# load the dataset
dataset0 = pd.read_csv('data0.csv')
dataset1 = pd.read_csv('data1.csv')
dataset2 = pd.read_csv('data_2.csv')

# split into input (X) and output (y) variables
data0=array(dataset0)
data1=array(dataset1)
data2=array(dataset2)

X0 = data0[:1400,:-1]
X1 = data1[:1400,:-1]
X2 = data2[:1400,:-1]



# number of signal points
# sample spacing
T = 1.0 / 9600.0

# number of signal points
# sample spacing
T = 1.0 / 9600.0
X0 = X0.reshape(-1)
X1 = X1.reshape(-1)
X2 = X2.reshape(-1)

N = len(X1)
x = np.linspace(0.0, N*T, N, endpoint=False)
sos = signal.iirfilter(88, [50, 500], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')


F0 = signal.sosfilt(sos,X0)
F1 = signal.sosfilt(sos,X1)
F2 = signal.sosfilt(sos,X2)




#ALL Filtering 
fig7,(ax1) = plt.subplots(1, 1)
fig7.suptitle('EMG signal Time & Frequency Domain 50,400')
ax1.plot(X0, label='X0')
ax1.plot(X1, label='X1')
ax1.plot(X2, label='X2')


ax1.plot(F0,label='Filter X0')
ax1.plot(F1,label='Filter X1')
ax1.plot(F2,label='Filter X2')


ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')
plt.legend()
plt.grid()
plt.show()
