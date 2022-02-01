#! /usr/bin/env python3
#
# Compute a random spectrum,
# then hand calculate a windowed FFT and Welch's PSD
# and compare it to Scipy's Welch's method
#
# Feb-2022, Pat Welch

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy import signal
import time
import sys

def calcPSD(xFFT:np.array, yFFT:np.array, fs:float, window:str="boxcar", n:int=None) -> np.array:
    if n is None:
        nfft = xFFT.size
        qOdd = nfft % 2 # Odd number of bins in FFT
        n = (nfft - qOdd) * 2 # Number of samples that went into the FFT
    w = Bias(n, window)
    wSum = (w * w).sum()
    psd = (xFFT.conjugate() * yFFT) / (fs * wSum)
    if not qOdd:        # Odd number of FFT bins
        psd[1:] *= 2    # Real FFT -> double for non-zero freq
    else:               # last point unpaired in Nyquist freq
        psd[1:-1] *= 2  # Real FFT -> double for non-zero freq
    return psd

def Bias(N:int, window="hann") -> np.array:
    if window == "boxcar": return np.ones(N)
    if window == "hann": 
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(0, N) / N))
    raise NotImplemented

def wfft(data:np.array, width:int, window:str="hann", qOverlap:bool=True) -> list[np.array]:
    bias = Bias(width, window)

    ffts = []
    for i in range(0,data.size - width + 1, (width // 2) if qOverlap else width):
        w = data[i:i+width]
        ffts.append(np.fft.rfft(w * bias))
    return ffts

def wcalcPSD(
        A_FFT_windows:list[np.array],
        B_FFT_windows:list[np.array],
        fs:float,
        window:str) -> np.array:
    width = A_FFT_windows[0].size
    spectrums = np.complex128(np.zeros(width))
    bias = Bias(width, "hann")
    for i in range(len(A_FFT_windows)):
        A = A_FFT_windows[i]
        B = B_FFT_windows[i]

        spectrum = calcPSD(A, B, fs=fs, window=window)
        spectrums += spectrum
    return spectrums / len(A_FFT_windows)

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=round(time.time()), help="Random number seed")
parser.add_argument("--window", type=str, default="boxcar",
        choices=("boxcar", "hann"),
        help="Window to test")
parser.add_argument("--fs", type=float, default=2, help="Sampling frequency")
parser.add_argument("--n", type=int, default=8192, help="Number of samples")
parser.add_argument("--width", type=int, default=512, help="Window width")
parser.add_argument("--nooverlap", action="store_true", help="Don't use overlapping windows")
args = parser.parse_args()

if args.n is None:
    args.n = np.power(2, np.arange(6, 14))

print("Random Seed", args.seed)
rng = np.random.default_rng(args.seed)
data = 2 * (rng.random(args.n) - 0.5) # Time domain data sample [-1,1)

ffts = wfft(data, args.width, qOverlap=(not args.nooverlap), window=args.window)
psd0 = wcalcPSD(ffts, ffts, args.fs, args.window)
(f, psd1) = signal.welch(data,
        fs=args.fs,
        window=args.window,
        nperseg=args.width,
        noverlap=0 if args.nooverlap else None,
        )

print(psd0.real - psd1)
print(psd0[0].real, psd1[0])
print(psd0[-1].real, psd1[-1])
