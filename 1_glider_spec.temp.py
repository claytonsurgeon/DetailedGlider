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
# Real Data
##################################
acct, accx, accy, accz = parse_csv()

acct = acct[2473:7083]
accx = accx[2473:7083]
accy = accy[2473:7083]
accz = accz[2473:7083]
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

# mock wave
A = np.fft.rfft(wave)
spectrum = normalize(A, wave, 'power', sampling_freq, samples)
# spectrum = normalize(A, wave, 'amplitude', sampling_freq, samples)
freq_space = np.fft.rfftfreq(n=samples, d=sampling_freq)

# real data
# print(len(accx))
real_sample_freq = acct[len(acct)-1]-acct[0]/len(acct)
A = np.fft.rfft(accx)
realspectrum = normalize(A, np.array(accx), 'amplitude', real_sample_freq, len(acct))
real_freq_space = np.fft.rfftfreq(n=len(accx), d=real_sample_freq)

A = np.fft.rfft(accy)
realspectrum2 = normalize(A, np.array(accy), 'amplitude', real_sample_freq, len(acct))

A = np.fft.rfft(accz)
realspectrum3 = normalize(A, np.array(accz), 'amplitude', real_sample_freq, len(acct))



##################################
# Welch method using Hann windows
##################################
# calculate number of windows and samples in each window
num_of_windows = samples // 2**8
M = samples // num_of_windows 
final_thing = windowfft("boxcar", samples, M, num_of_windows, wave, sampling_freq)
freq_space_window = np.fft.rfftfreq(n=M, d=sampling_freq)


num_of_windows = 6
M = len(accx) // num_of_windows

# A = np.fft.rfft(accx)
# realspectrum = windowfft("hann", len(accx), M, num_of_windows, accx, real_sample_freq)
# real_freq_space = np.fft.rfftfreq(n=M, d=real_sample_freq)

# A = np.fft.rfft(accy)
# realspectrum2 = windowfft("hann", len(accy), M, num_of_windows, accy, real_sample_freq)


# A = np.fft.rfft(accz)
# realspectrum3 = windowfft("hann", len(accz), M, num_of_windows, accz, real_sample_freq)


##################################
# Calculate Banding
##################################


##################################
# Calculating significant wave height
##################################
a0 = getSWH(spectrum)
print(a0)

##################################
# Calculate PeakPSD
##################################
print("peakPSD without windowing = ", spectrum.max())
print("peakPSD with windowing =    ", final_thing.max())
    
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

# real_sample_freq = acct[len(acct)-1]-acct[0]/len(acct)
# A = np.fft.rfft(accx)
# realspectrum = normalize(A, accx, 'power', real_sample_freq, len(acct))
# real_freq_space = np.fft.rfftfreq(n=len(accx), d=real_sample_freq)


fig2, [acc1, acc2, acc3, fft1, fft2, fft3] = plt.subplots(nrows = 6, ncols= 1)
acc1.plot(acct, accx)
acc2.plot(acct, accy)
acc3.plot(acct, accz)
fft1.plot(real_freq_space, realspectrum)
fft2.plot(real_freq_space, realspectrum2)
fft3.plot(real_freq_space, realspectrum3)

acc1.set_title("accX")
fft1.set_title("accX FFT")
acc1.set_xlabel("time (second)")
acc1.set_ylabel("m/s^2")

acc2.set_title("accY")
fft2.set_title("accy FFT")
acc2.set_xlabel("time (second)")
acc2.set_ylabel("m/s^2")

acc3.set_title("accZ")
fft3.set_title("accz FFT")
acc3.set_xlabel("time (second)")
acc3.set_ylabel("m/s^2")
plt.tight_layout()
plt.show()



