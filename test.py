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
dataset = pd.read_csv('Emg1.csv')
# split into input (X) and output (y) variables
data=array(dataset)
X1 = data[:4766,:-1]
y = data[:4766,-1:]
# number of signal points
# sample spacing
T = 1.0 / 9600.0

# number of signal points
# sample spacing
T = 1.0 / 9600.0
X1 = X1.reshape(-1)
N = len(X1)
x = np.linspace(0.0, N*T, N, endpoint=False)
sos = signal.iirfilter(91, [50, 400], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
sos1 = signal.iirfilter(91, [50, 450], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
sos2 = signal.iirfilter(91, [50, 500], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
sos3 = signal.iirfilter(91, [50, 1000], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
sos4 = signal.iirfilter(91, [1000, 1500], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
sos5= signal.iirfilter(91, [50, 300], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
F1 = signal.sosfilt(sos,X1)
F2 = signal.sosfilt(sos1,X1)
F3 = signal.sosfilt(sos2,X1)
F4 = signal.sosfilt(sos3,X1)
F5 = signal.sosfilt(sos4,X1)
F6 = signal.sosfilt(sos5,X1)
yf0 = fft(X1)
xf = fftfreq(N, T)
xf = fftshift(xf)
yplot0 = fftshift(yf0)
#F1
yf1 = fft(F1)
yplot1 = fftshift(yf1)
#F2
yf2 = fft(F2)
yplot2 = fftshift(yf2)
#F3
yf3 = fft(F3)
yplot3 = fftshift(yf3)
#F4
yf4 = fft(F4)
yplot4 = fftshift(yf4)
#F4
yf5 = fft(F5)
yplot5 = fftshift(yf5)

yf6 = fft(F6)
yplot6 = fftshift(yf6)
#orginal time & Frequency Domain
fig1,(ax1) = plt.subplots(1, 1)
fig1.suptitle('EMG signal Time & Frequency Domain')
ax1.plot(X1, 'g-')
ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')


#Filtering 50 - 2000 Hz time & Frequency Domain
fig2,(ax1) = plt.subplots(1, 1)
fig2.suptitle('EMG signal Time & Frequency Domain After Filter : 50 - 2000 Hz ')
ax1.plot(F5, 'g-')
ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')


#Filtering 50 - 1000 Hz time & Frequency Domain
fig3,(ax1) = plt.subplots(1, 1)
fig3.suptitle('EMG signal Time & Frequency Domain After Filter : 50 - 1000 Hz ')
ax1.plot(F4, 'g-')
ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')


#Filtering 50 - 500 Hz time & Frequency Domain
fig4,(ax1) = plt.subplots(1, 1)
fig4.suptitle('EMG signal Time & Frequency Domain After Filter : 50 - 500 Hz ')
ax1.plot(F3, 'g-')
ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')


#Filtering 50 - 450 Hz time & Frequency Domain
fig5,(ax1) = plt.subplots(1, 1)
fig5.suptitle('EMG signal Time & Frequency Domain After Filter : 50 - 450 Hz ')
ax1.plot(F2, 'g-')
ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')


#Filtering 50 - 400 Hz time & Frequency Domain
fig6,(ax1) = plt.subplots(1, 1)
fig6.suptitle('EMG signal Time & Frequency Domain After Filter : 50 - 400 Hz ')
ax1.plot(F1, 'g-')
ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')


#ALL Filtering 
fig7,(ax1) = plt.subplots(1, 1)
fig7.suptitle('EMG signal Time & Frequency Domain')
ax1.plot(X1, label='Original')
ax1.plot(F5,label='Filter 1000-1500 Hz')
# ax1.plot(F4,label='Filter 50-1000 Hz')
ax1.plot(F6,label='Filter 50-300 Hz')
ax1.plot(F2,label='Filter 50-450 Hz')
# ax1.plot(F1,label='Filter 50-400 Hz')
ax1.plot(y,label='Target')
ax1.set_ylabel('Emg Signal')
ax1.set_xlabel('Sample')
plt.legend()
plt.grid()
plt.show()
