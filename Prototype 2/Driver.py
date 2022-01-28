from random import sample
from time import time
from xmlrpc.client import UNSUPPORTED_ENCODING
import matplotlib.pyplot as plt
import numpy as np
from helpers import *

"""
-----------------------------------------------------------
-----------------------------------------------------------
Read Data file

"""

##################################
# Real Data
##################################
real_data = parse_csv('SampleData\Output.csv')
fs = parse_fs("meta.csv")
data = parse_ts("acceleration.csv")
freq_bound = parse_frequency("freq.csv")

#print("acceleration.csv:", ts)
#print("meta.csv:", fs)
#print("wave.cs:", time_bounds)

# Pat's data
acct = data.ts
accx = data.x
accy = data.y
accz = data.z

# Our data
'''
acct = real_data.t
accx = real_data.x
accy = real_data.y
accz = real_data.z
'''



# accx *= np.less(accx, 10)
cutoff = 10

for i in range(len(accx)):
    temp = accx[i]
    if abs(accx[i]) > cutoff:
        temp = 0
    accx[i]=temp

for i in range(len(accy)):
    temp = accy[i]
    if abs(accy[i]) > cutoff:
        temp = 0
    accy[i]=temp

for i in range(len(accz)):
    temp = accz[i]
    if abs(accz[i]) > cutoff:
        temp = 0
    accz[i]=temp


#sample_freq = ((acct[len(acct)-1]-acct[0])/len(acct))

freq_space = np.fft.rfftfreq(n=len(accx), d=fs)

#print("sample_freq = ", 1/sample_freq)


##################################
# normal fft
##################################

xFFT = np.fft.rfft(accx)
spectrumX = calcPSD(xFFT, xFFT, fs).real


##################################
# Welch method using Hann windows
##################################

num_of_windows = 6
M = len(acct) // num_of_windows
spectrumXW = windowfft(accx, num_of_windows, fs, "hann")
freq_space_window = np.fft.rfftfreq(n=M, d=fs)


##################################
# Calculate Banding
##################################

# upper and lower bound of frequency
# [:] to hold all their values
fLower = freq_bound.fLower[:]
fUpper = freq_bound.fUpper[:]

# forcing them to be a 2D numpy array
temp_lower = np.asarray(fLower)
temp_upper = np.asarray(fUpper)
freqBound = np.column_stack((temp_lower, temp_upper))
f = np.fft.rfftfreq(len(data.z), 1/fs) # Pat's example of f


# the quantile according to Pat
quant = np.logical_and(
        np.less_equal.outer(freqBound[:,0], f),
        np.greater_equal.outer(freqBound[:,1], f)
        ) 

cnt = quant.sum(axis=1)

# .sum(axis=1) is a summation of columns
# .sum(axis=0) is a summation of rows
xxBand = (quant * spectrumX).sum(axis=1) / cnt

print(xxBand)
exit(0)


##################################
# Calculating significant wave height
##################################

##################################
# Calculate PeakPSD
##################################

    
##################################
# Plotting
##################################

# real data plotting
fig1, [acc1, fft1] = plt.subplots(nrows = 2, ncols= 1)

acc1.plot(acct, accx)
acc1.set_title("accX")
acc1.set_xlabel("time (second)")
acc1.set_ylabel("m/s^2")


fft1.plot(freq_space, spectrumX)
fft1.set_title("accX FFT")
fft1.set_xlabel("Frequency (Hz)")
fft1.set_ylabel("PSD")
fft1.set_xlim(0.03, 0.5)
plt.tight_layout()


fig2, [acc1W, fft1W] = plt.subplots(nrows = 2, ncols= 1)

acc1W.plot(acct, accx)
acc1W.set_title("accX")
acc1W.set_xlabel("time (second)")
acc1W.set_ylabel("m/s^2")


fft1W.plot(freq_space_window, spectrumXW)
fft1W.set_title("accX FFT")
fft1W.set_xlabel("Frequency (Hz)")
fft1W.set_ylabel("PSD")
fft1W.set_xlim(0.03, 0.5)


plt.tight_layout()
plt.show()



