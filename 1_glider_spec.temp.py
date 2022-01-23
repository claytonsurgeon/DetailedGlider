from sunau import _sunau_params
import matplotlib.pyplot as plt
import numpy as np
from helpers import *

"""
-----------------------------------------------------------
-----------------------------------------------------------
Read Data file

"""


##################################
# Mock Wave
##################################
duration = 20*60    # 20 minute sampling period
samples  = 2**12    # number of samples in data set
sampling_freq = duration / (samples - 1)

# array of time increments
time = np.linspace(0, duration, samples, endpoint=False)

# a composite wave for basic testing
# let's pretend this is eastward acceleration data
wave = 10*np.sin(2*np.pi*time * (1/1)) + 2*np.cos(2*np.pi*time * (1/30))

##################################
# normal fft
##################################
A = np.fft.rfft(wave)
spectrum = normalize(A, wave, 'power', sampling_freq, samples)
# spectrum = normalize(A, wave, 'amplitude', sampling_freq, samples)


freq_space = np.fft.rfftfreq(n=samples, d=sampling_freq)


##################################
# Welch method using Hann windows
##################################
# calculate number of windows and samples in each window
num_of_windows = samples // 2**8
M = samples // num_of_windows 
final_thing = windowfft("hann", samples, M, num_of_windows, wave, sampling_freq)
freq_space_window = np.fft.rfftfreq(n=M, d=sampling_freq)


##################################
# Calculating significant wave height
##################################
a0 = getSWH(spectrum)
print(a0)
    


# plotting
fig, [ax1, ax2, ax3] = plt.subplots(nrows = 3, ncols= 1)
ax1.plot(time, wave)
ax2.plot(freq_space, spectrum, '.-')
ax3.plot(freq_space_window, final_thing, '.-')


ax1.set_title("Original Wave")
ax1.set_xlabel("time (second)")
ax1.set_ylabel("m/s^2")

ax2.set_title("Normal FFT")
ax2.set_xlabel("Freq (Hz)")
ax2.set_ylabel("PSD")

ax3.set_title("Hann winodow FFT")
ax3.set_xlabel("Freq (Hz)")
ax3.set_ylabel("PSD")



ax1.grid()
ax2.grid()
ax3.grid()


ax1.set_xlim(0, time[-1])  #sets a limit for x
ax2.set_xlim(0, freq_space[-1]) 
ax3.set_xlim(0, freq_space_window[-1]) 

plt.tight_layout() #fix titles being cut off
plt.show()
