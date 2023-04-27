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

data = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Kwan/fist_hand/1.csv')
data0 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Kwan/open_hand/1.csv')
data1 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Kwan/love_hand/1.csv')
data2 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Kwan/two_finger/1.csv')
data3 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Liu/two_finger/1t.csv')
data4 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Liu/love_hand/1l.csv')
data5 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Liu/fist_hand/1f.csv')
data6 = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Two_sensor/Liu/open_hand/1.csv')

X6 = data6.iloc[:,:1]
X = data.iloc[:,:1]
X0 = data0.iloc[:,:1]
X1 = data1.iloc[:,:1]
X2 = data2.iloc[:,:1]
X3 = data3.iloc[:,:1]
X4 = data4.iloc[:,:1]
X5 = data5.iloc[:,:1]
import matplotlib.pyplot as plt
fig,(ax1) = plt.subplots(1, 1)


ax1.plot(X5, label='Column 2', color='blue')
# ax2.plot(X1, label='Column 2', color='pink')
# ax3.plot(X0, label='Column 2', color='green')
# ax4.plot(X2, label='Column 2', color='red')
plt.grid()
plt.show()
