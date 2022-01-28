from cmath import nan
import datetime
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
wave_data = parse_wave_data("wave.csv")

start_time = datetime.datetime.strptime(str( wave_data.tLower[0])[:-3], '%Y-%m-%d %H:%M').timestamp()


#print("acceleration.csv:", ts)
#print("meta.csv:", fs)
#print("wave.cs:", time_bounds)

# Pat's data
acct = data.ts
accx = data.x
accy = data.y
accz = data.z


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

# print(acct)

j = 0
blocks = []
for i in range(len(wave_data.tLower)):
    block = [[], [], [], []]
    start = datetime.datetime.strptime(str( wave_data.tLower[i])[:-3], '%Y-%m-%d %H:%M').timestamp() - start_time
    end = datetime.datetime.strptime(str( wave_data.tUpper[i])[:-3], '%Y-%m-%d %H:%M').timestamp() - start_time
    # print("Block [", i, "] = ", "timerange (s) = ", start , " to ", end)
    
    difference = np.absolute(np.array(acct)-start)
    j = difference.argmin()

    while start < end: 
        block[0].append(acct[j]) 
        # print("j = ", j, " accx[j] = ", accx[j])
        block[1].append(accx[j])
        block[2].append(accy[j])
        block[3].append(accz[j])
        start += 1/fs
        j+=1
    blocks.append(block)




# Our data
'''
acct = real_data.t
accx = real_data.x
accy = real_data.y
accz = real_data.z
'''


block_selector = 1



#sample_freq = ((acct[len(acct)-1]-acct[0])/len(acct))

freq_space = np.fft.rfftfreq(n=len(blocks[block_selector][1]), d=fs)



#print("sample_freq = ", 1/sample_freq)


##################################
# normal fft
##################################

xFFT = np.fft.rfft(blocks[block_selector][1])
yFFT = np.fft.rfft(blocks[block_selector][2])
zFFT = np.fft.rfft(blocks[block_selector][3])

spectrumX = calcPSD(xFFT, xFFT, fs).real
spectrumY = calcPSD(yFFT, yFFT, fs).real
spectrumZ = calcPSD(zFFT, zFFT, fs).real



##################################
# Welch method using Hann windows
##################################
M = 2**10

num_of_windows = len(blocks[block_selector][0]) // M 

print("num_of_windows = ", num_of_windows)
print("M = ", M)

freq_space_window = np.fft.rfftfreq(n=M, d=fs) # compatible with Welch Method

spectrumXW = windowfft(blocks[block_selector][1], M, fs, "hann")

spectrumYW = windowfft(blocks[block_selector][2], M, fs, "hann")

spectrumZW = windowfft(blocks[block_selector][3], M, fs, "hann")



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


f = np.fft.rfftfreq(len(blocks[block_selector][3]), 1/fs) # compatitble with Normal FFT

# the quantile according to Pat
quant = np.logical_and(
        np.less_equal.outer(freqBound[:,0], f),
        np.greater_equal.outer(freqBound[:,1], f)
        ) 

# [True false True] = 2
cnt = quant.sum(axis=1)

print(cnt)




# .sum(axis=1) is a summation of columns
# .sum(axis=0) is a summation of rows

xxBand = (quant * spectrumX).sum(axis=1) / cnt
yyBand = (quant * spectrumY).sum(axis=1) / cnt
zzBand = (quant * spectrumZ).sum(axis=1) / cnt

# print(xxBand)



##################################
# Calculating significant wave height
##################################

fMid = freqBound.mean(axis=1)


a0 = zzBand / np.square(np.square(2*np.pi*fMid))

m0 = (a0 * freq_bound.bandwidth).sum()

# print("fmid = ", fMid)
# print("a0 = ", a0)
print("m0 = ", 4 * np.sqrt(m0), "\texpected m0 = ", wave_data.Hs[block_selector])


##################################
# Calculate PeakPSD
##################################

    
##################################
# Plotting
##################################




# real data plotting
fig1, [acc1, fft1] = plt.subplots(nrows = 2, ncols= 1)

acc1.plot(blocks[block_selector][0], blocks[block_selector][1])
acc1.set_title("accX")
acc1.set_xlabel("time (second)")
acc1.set_ylabel("m/s^2")


fft1.plot(freqBound, xxBand)
fft1.set_title("accX FFT")
fft1.set_xlabel("Frequency (Hz)")
fft1.set_ylabel("PSD")
fft1.set_xlim(0.03, 0.5)
plt.tight_layout()


fig2, [acc1W, fft1W] = plt.subplots(nrows = 2, ncols= 1)

acc1W.plot(blocks[block_selector][0], blocks[block_selector][1])
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



