import matplotlib.pyplot as plt
import numpy as np
from helpers import *

"""
-----------------------------------------------------------
-----------------------------------------------------------
Read Data file

"""


"""
-----------------------------------------------------------
-----------------------------------------------------------
Mock Wave

"""

# hann_window = 0.5(1-cos(2*pi*x / (M-1)))


duration = 20*60    # 20 minute sampling period
samples  = 2**10     # number of samples in data set

# array of time increments
time = np.linspace(0, duration, samples, endpoint=False)
# print(time)

# a composite wave for basic testing
# let's pretend this is eastward acceleration data
wave = 10*np.sin(2*np.pi*time * (1/100)) + 2*np.cos(2*np.pi*time * (1/200))


M = 2**7
num_of_windows = samples // M - 1
print(f"num_of_windows {num_of_windows}")
hann_window = 0.5*(1 - np.cos(2*np.pi*time[0:M] / (M - 0)))
window1 = wave[0:M] * hann_window

windows = [window1]
counter = 0
while counter < num_of_windows:
	# print(counter*M+(M//2), counter*M+(M//2)+M)
	next_window = wave[counter*M+(M//2):counter*M+(M//2)+M] * hann_window
	windows.append(next_window)
	counter += 1 
print(len(windows))
	

# exit(0)
print(wave)

sampling_freq = duration / (samples - 0)


# display the original wave time vs m/s^2
plt.figure(1)
plt.plot(time, wave)
plt.xlim(0, duration)
plt.xlabel("time (second)")
plt.ylabel("m/s^2")
plt.title('Original Signal in Time Domain')


"""
-----------------------------------------------------------
-----------------------------------------------------------
Intial Data smoothing

"""


"""
-----------------------------------------------------------
-----------------------------------------------------------
FFT

"""


# get the magnitudee of the spectrum then normalize by number of 'buckets' in the spectrum
spectrum = np.abs(np.fft.rfft(wave)[0:-1]) / (sampling_freq*samples)
# spectrum = np.abs(np.fft.fft(wave)) / (samples/2)

# an array of frequency increments
freq_space = np.fft.fftfreq(n=samples, d=sampling_freq)[0:samples//2]
# freq_space = np.fft.fftfreq(n=samples, d=sampling_freq)[0:samples]

"""
From Pat:
The normalization to get (m/s^2)^2/Hz is

Divide everything by sampling frequency * number of bins. For a 2048 sample size, at 1.71 samples/sec the normalization would be
1.71 * 2048

All bins, except the f=0 and potentially the f=f_nyquist bins are doubled due to the symmetry of positive and negative frequencies. i.e. the power is split between the positive and negative frequency bins, and they are the same. So you multiply by 2.

For 2048 sample bins, your FFT will have 1025 frequencies. f[0] is zero, the DC component. f[-1] is the f_nyquist frequency which does not have a matching component, so the doubling would be

PSD[1:-1] *=2
"""


plt.figure(2)
plt.plot(freq_space, spectrum)
plt.ylabel("Amplitude, m/s^2")
plt.xlabel("freq (Hz)")
plt.title('freq Domain')


displacement = list(spectrum)

#loop through data and apply the displacement spectrum density conversion to every PSD value. 
for i in range(len(displacement)): 
	if(spectrum[i] > 0):
		FAS = spectrum[i]
		displacement[i] = FAS/(math.pow((2*math.pi*freq_space[i]), 2))


plt.figure(3)
plt.plot(freq_space, displacement)
plt.ylabel("Amplitude, m^2/Hz")
plt.xlabel("freq (Hz)")
plt.title('freq Domain')

plt.show()
