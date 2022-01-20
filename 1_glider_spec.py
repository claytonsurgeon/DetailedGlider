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

"""
-----------------------------------------------------------
-----------------------------------------------------------
Notes
hann_window = 0.5(1-cos(2*pi*x / (M-1)))

"""

#setting up perameters
Sfreq = 2000 #sampling frequency
Tstep = 1/Sfreq #sample time interval

freq1 = 1/10 #frequency of wave 1
mag1 = 10 #amplitude of wave 1

freq2 = 1/15 #frequency of wave 2
mag2 = 2 #amplitude of wave 2

sample_scaler = 20

# number of samples in data set
samples  = int(sample_scaler * Sfreq / freq1)     

# 20 minute sampling period
duration = (samples-1)*Tstep    

# array of time increments
time = np.linspace(0, duration, samples)
fstep = Sfreq / samples # freq interval
f = np.linspace(0, (samples-1) * fstep, samples) #freq steps

# a composite wave for basic testing
wave = mag1*np.sin(2 * np.pi * time * freq1) + mag2 * np.cos(2 * np.pi * time * freq2)

# FFT
spectrum = np.fft.fft(wave)
spectrum_mag = np.abs(spectrum) / samples

#max wave is half of Sampling frequency. 
f_plot = f[0:int(samples/2+1)]
x_mag_plot = 2 * spectrum_mag[0:int(samples/2+1)]
x_mag_plot[0] = spectrum_mag[0] / 2 

#loop through data and apply the displacement spectrum density conversion to every PSD value. 
displacement = list(x_mag_plot)
print(len(f_plot), " ", len(x_mag_plot))

for i in range(len(displacement)): 
	if(spectrum[i] > 0):
		FAS = spectrum[i]
		displacement[i] = FAS/(math.pow((2 * math.pi * x_mag_plot[i]), 2))

"""
hann windows/Welch method
"""
M=0
hann_window = 0.5*(1 - np.cos(2*np.pi*np.array(range(0, M)) / (M - 0)))


"""
From Pat:
The normalization to get (m/s^2)^2/Hz is

Divide everything by sampling frequency * number of bins. For a 2048 sample size, at 1.71 samples/sec the normalization would be
1.71 * 2048

All bins, except the f=0 and potentially the f=f_nyquist bins are doubled due to the symmetry of positive and negative frequencies. i.e. the power is split between the positive and negative frequency bins, and they are the same. So you multiply by 2.

For 2048 sample bins, your FFT will have 1025 frequencies. f[0] is zero, the DC component. f[-1] is the f_nyquist frequency which does not have a matching component, so the doubling would be

PSD[1:-1] *=2
"""


# plotting
fig, [ax1, ax2, ax3] = plt.subplots(nrows = 3, ncols= 1)
ax1.plot(time, wave)
ax2.plot(f_plot, x_mag_plot, '.-')
ax3.plot(f_plot[0:len(displacement)], displacement, '.-')


ax1.set_xlabel("time (s)")

ax2.set_xlabel("Freq (Hz)")
ax2.set_ylabel("PSD")

ax3.set_xlabel("Freq (Hz)")
ax3.set_ylabel("PSD")



ax1.grid()
ax2.grid()
ax3.grid()


ax1.set_xlim(0, time[-1])  #sets a limit for x
ax2.set_xlim(0, f_plot[-1]) 
ax3.set_xlim(0, f_plot[-1]) 

plt.tight_layout() #fix titles being cut off
plt.show()

