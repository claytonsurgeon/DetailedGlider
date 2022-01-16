import matplotlib.pyplot as plt
import numpy as np

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

duration = 20*60    # 20 minute sampling period
samples = 2**12     # number of samples in data set

# array of time increments
time = np.linspace(0, duration, samples, endpoint=False)

# a composite wave for basic testing
# let's pretend this is eastward acceleration data
wave = 10*np.sin(2*np.pi*time * (1/100)) + 2*np.cos(2*np.pi*time * (1/200))


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
spectrum = np.abs(np.fft.rfft(wave)[0:-1]) / (samples/2)

# an array of frequency increments
freq_space = np.fft.fftfreq(n=samples, d=sampling_freq)[0:samples//2]

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

plt.show()
