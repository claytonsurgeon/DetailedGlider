from sunau import _sunau_params
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
samples  = 2**12    # number of samples in data set
sampling_freq = duration / (samples - 1)

# array of time increments
time = np.linspace(0, duration, samples, endpoint=False)


# a composite wave for basic testing
# let's pretend this is eastward acceleration data
wave = 10*np.sin(2*np.pi*time * (1/1)) + 2*np.cos(2*np.pi*time * (1/30))

"""
-----------------------------------------------------------
-----------------------------------------------------------
Normal FFT

"""

# get the magnitudee of the spectrum then normalize by number of 'buckets' in the spectrum
# spectrum = np.abs(np.fft.rfft(wave)[0:-1]) / (sampling_freq*samples)
# spectrum = np.abs(np.fft.rfft(wave)[0:-1]) / (samples/2)
A = np.fft.rfft(wave)
spectrum = (A.conjugate() * A).real / (wave.size * sampling_freq)
spectrum[1:-1] *= 2
# spectrum = 2 / sampling_freq * np.sqrt(spectrum * (wave.size * sampling_freq / 2))

#calculate amplitude
# spectrum = np.sqrt(A.conjugate()*A) / (samples/2)
# spectrum = np.abs(np.fft.rfft(wave)) / (samples/2)



# an array of frequency increments
freq_space = np.fft.rfftfreq(n=samples, d=sampling_freq)
# freq_space = np.fft.fftfreq(n=samples, d=sampling_freq)[0:samples]

print(len(freq_space), len(spectrum))

"""
-----------------------------------------------------------
-----------------------------------------------------------
Welch method using Hann windows

"""
# calculate number of windows and samples in each window
num_of_windows = samples // 2**8
M = samples // num_of_windows

#create a hann window
hann_window = 0.5*(1 - np.cos(2*np.pi*np.array(range(0, M)) / (M - 0)))
# hann_window = np.ones(M)

#fill 
window1 = wave[0:M] #* hann_window



windows = []
counter = 0
N = M//2

for i in range(0, samples-M+1, N):
	next_window = wave[i:i+M] #* hann_window
	windows.append(next_window)




# while counter < num_of_windows -1:
# 	# next_window = wave[counter*M+(M//2):counter*M+(M//2)+M] * hann_window
# 	# next_window = wave[counter*(M//2)+(M//2):counter*(M//2)+(M//2)+(M//2)] #* hann_window
	
# 	# N = M//2
# 	next_window = wave[counter*N+N:(counter+1)*N+N] * hann_window

# 	windows.append(next_window)
# 	counter += 1 



# print(hann_window.sum(), len(hann_window))

freq_space_window = np.fft.rfftfreq(n=M, d=sampling_freq)

spectrums = np.zeros(N+1)

denominator = hann_window.sum() * sampling_freq
for window in windows:
	# window *= hann_window
	
	A = np.fft.rfft(window*hann_window)
	spectrum1 = (A.conjugate() * A).real / denominator
	spectrum1[1:-1] *= 2
	# A = np.fft.rfft(wave)
	# spectrum = A.conj() * A / (wave.size * sampling_freq)
	# spectrum[1:-1] *= 2
	# spectrum1 = np.abs(np.fft.rfft(window)) / (M/2)
	# print(len(window), len(spectrum1))
	# spectrums.append(np.array(getDS(spectrum, freq_space_window)))
	spectrums += spectrum1

# temp = spectrums[0]
# for spectrum2 in spectrums[1:]: 
# 	temp += spectrum2

final_thing = spectrums/len(windows) * num_of_windows
    




"""
From Pat:
The normalization to get (m/s^2)^2/Hz is

Divide everything by sampling frequency * number of bins. For a 2048 sample size, at 1.71 samples/sec the normalization would be
1.71 * 2048

All bins, except the f=0 and potentially the f=f_nyquist bins are doubled due to the symmetry of positive and negative frequencies. i.e. the power is split between the positive and negative frequency bins, and they are the same. So you multiply by 2.

For 2048 sample bins, your FFT will have 1025 frequencies. f[0] is zero, the DC component. f[-1] is the f_nyquist frequency which does not have a matching component, so the doubling would be

PSD[1:-1] *=2
"""


# display the original wave time vs m/s^2
# plt.figure(1)
# plt.plot(time, wave)
# plt.xlim(0, duration)
# plt.xlabel("time (second)")
# plt.ylabel("m/s^2")
# plt.title('Original Signal in Time Domain')


# plt.figure(2)
# plt.plot(freq_space, spectrum)
# plt.ylabel("Amplitude, m/s^2")
# plt.xlabel("freq (Hz)")
# plt.title('freq Domain')

# plt.figure(4)
# plt.plot(freq_space_window, final_thing)
# plt.ylabel("Amplitude, m^2/Hz")
# plt.xlabel("freq (Hz)")
# plt.title('freq Domain')

# plt.show()



# plotting
fig, [ax1, ax2, ax3] = plt.subplots(nrows = 3, ncols= 1)
ax1.plot(time, wave)
print(len(freq_space), len(spectrum))
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
