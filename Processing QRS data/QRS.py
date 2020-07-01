import os
import sys
import math
import numpy as np
import pandas as pd
from time import gmtime, strftime
from scipy.signal import butter, lfilter, iirnotch
import matplotlib.pyplot as plt

 # NOTCH FILTER #


def notch_filter(data, fs, fnotch):
    nyq = 0.5 * fs  # Nyquist frequeny is half the sampling frequency
    w0 = fnotch / nyq
    b, a = iirnotch(w0, 30.0)
    data_filtered = lfilter(b, a, data)
    return data_filtered


# Bandpass filter FILTER #


def BP_filter(data, fs):
    nyq = 0.5 * fs
    low = 0.1 / nyq
    high = 45.0 / nyq
    b, a = butter(2, [low, high], btype='band')
    data_filtered = lfilter(b, a, data)
    return data_filtered

# Method to find the peak


def get_peak_index(data, spacing, limit):
    length = data.size
    x = np.zeros(length + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + length] = data
    peak_candidate = np.zeros(length)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + length]  # before
        start = spacing
        h_c = x[start: start + length]  # central
        start = spacing + s + 1
        h_a = x[start: start + length]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


# Reading Data and filtering

window = 25

Dataset = np.loadtxt('DataN.txt')

Notch_filtered = notch_filter(Dataset, 256, 50) # Signal with notch filter
BP_filtered = BP_filter(Notch_filtered, 256)    # Signal with notch filter + BPF
differentiated_signal = np.ediff1d(BP_filtered)  # Differentiated signal
squared_signal = differentiated_signal ** 2      # Sqauring the signal
Signal_final = np.convolve(squared_signal, np.ones(window))  # Moving-window integration.

# Peak detecting

#peak_index = get_peak_index(Dataset,50, 0.05)
#peak_value = Dataset[peak_index]


peak_index = np.array([], dtype=int)
peak_index = get_peak_index(Signal_final, 50, 0.05) #filtered
peak_value = Signal_final[peak_index]

i=0
RR = []
while (i< (len(peak_index) -1)):
    RR_interval = (peak_index[i+1] - peak_index[i]) #Calculate distance between beats in # of samples
    ms_dist = ((RR_interval / 256) * 1000.0) #Convert sample distances to ms distances
    RR.append(ms_dist)
    i= i+1



# Plotting

plt.subplot(211)
plt.title("Data before filtering")
plt.plot(Dataset, color='Red')
plt.xlim(0, 2000)
plt.title("Detected peaks in signal,N=25")
plt.subplot(212)
plt.plot(Signal_final, color='Green')
plt.scatter(peak_index, peak_value, color='Black', marker='*')
plt.xlim(0, 2000)
# plt.title("RR_interval in ms")  # plot RR interval
# plt.plot(RR, color='Black')
plt.xlim(0, 2000)
plt.show()