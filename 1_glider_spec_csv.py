from cmath import sqrt
from sunau import _sunau_params
from typing import final
import matplotlib.pyplot as plt
import numpy as np
import csv
from helpers import *

class Direction:
    x = []
    y = []
    z = []

class Calculation:
    duration = 20*60 # 20 minutes
    samples = 2**12 # 4096 samples
    fs = duration / (samples - 1) # frequency samples
    time = np.linspace(0, duration, samples, endpoint=False) # time
    wave = 10*np.sin(2*np.pi*time * (1/1)) + 2*np.cos(2*np.pi* time * (1/30)) # mock wave
    freq_space = np.fft.rfftfreq(n=samples, d=fs) 
    num_of_windows = samples // 2**8
    M = samples // num_of_windows
    hann_window = 0.5*(1 - np.cos(2*np.pi*np.array(range(0, M)) / (M - 0)))


def parse_csv():
    with open('station222.csv', newline='') as csvfile:
        #https://docs.python.org/3/library/csv.html
        #https://evanhahn.com/python-skip-header-csv-reader/
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        for row in spamreader:
            # According to the csv file
            Direction.x.append((float(row[1])))
            Direction.y.append((float(row[2])))
            Direction.z.append((float(row[0])))

        #print(Directions.x)
        #print(Directions.y)
        #print(Directions.z)


def calcPSD(xAxis, yAxis):
    windows = []
    N = Calculation.M//2

    for i in range(0, Calculation.samples-Calculation.M+1, N):
        next_window = Calculation.wave[i:i+Calculation.M] 
        windows.append(next_window)

    spectrums = np.zeros(N+1)
    denominator = Calculation.hann_window.sum() * Calculation.fs

    #for window in windows:
    #    spectrum1 = (xAxis.conjugate() * yAxis).real / denominator
    #    spectrum1[1:-1] *= 2
    #    spectrums += spectrum1
    
    #PSD = spectrums/len(windows) * Calculation.num_of_windows
    #return PSD

    spectrum1 = (xAxis.conjugate() * yAxis).real / denominator
    spectrum1[1:-1] *= 2

    return spectrum1


def steps_toDO():
    x = np.array(Direction.x)
    y = np.array(Direction.y)
    z = np.array(Direction.z)
    
    xFFT = np.fft.rfft(x, n=z.size)
    yFFT = np.fft.rfft(y, n=z.size)
    zFFT = np.fft.rfft(z, n=z.size)
    #f = np.fft.rfftfreq(Direction.z, 1/Calculation.fs)

    xxPSD = calcPSD(xFFT, xFFT).real
    #xyPSD = calcPSD(xFFT, yFFT, Calculation.fs)

   # yyPSD = calcPSD(yFFT, yFFT, Calculation.fs).real

   # zxPSD = calcPSD(zFFT, xFFT, Calculation.fs)
   # zyPSD = calcPSD(zFFT, yFFT, Calculation.fs)
   # zzPSD = calcPSD(zFFT, zFFT, Calculation.fs).real

    print(xxPSD)

def main():
    parse_csv()
    steps_toDO()

if __name__ == '__main__':
    main()
    exit(0)

