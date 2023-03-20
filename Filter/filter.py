import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz, filtfilt

# Define functions for various filters

# Moving Average filter

# Butterworth low-pass filter
def butter_lowpass_filter(data, cutoff_freq, sample_freq, order):
    nyq_freq = 0.5 * sample_freq
    normal_cutoff = cutoff_freq / nyq_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Butterworth high-pass filter
def butter_highpass_filter(data, cutoff_freq, sample_freq, order):
    nyq_freq = 0.5 * sample_freq
    normal_cutoff = cutoff_freq / nyq_freq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Butterworth band-pass filter
def butter_bandpass_filter(data, lowcut_freq, highcut_freq, sample_freq, order):
    nyq_freq = 0.5 * sample_freq
    low = lowcut_freq / nyq_freq
    high = highcut_freq / nyq_freq
    b, a = butter(order, [50, 100], btype='band', analog=False)
    return filtfilt(b, a, data)

# Notch filter
def notch_filter(data, cutoff_freq, quality_factor, sample_freq):
    w0 = cutoff_freq / (sample_freq / 2)
    Q = quality_factor
    b, a = butter(2, [w0/Q, w0*Q], btype='bandstop')
    return filtfilt(b, a, data)

# Load data
data = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Data/Data12/All0.csv')

# Extract feature matrix X and target variable y
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
# Apply Moving Average filter to X


# Apply Butterworth low-pass filter to X
X_filtered2 = butter_lowpass_filter(X, cutoff_freq=5, sample_freq=1000, order=5)

# Apply Butterworth high-pass filter to X
X_filtered3 = butter_highpass_filter(X, cutoff_freq=30, sample_freq=1000, order=5)

# Apply Butterworth band-pass filter to X
X_filtered4 = butter_bandpass_filter(X, lowcut_freq=10, highcut_freq=50, sample_freq=1000, order=5)

# Apply Notch filter to X
X_filtered5 = notch_filter(X, cutoff_freq=60, quality_factor=30, sample_freq=1000)
