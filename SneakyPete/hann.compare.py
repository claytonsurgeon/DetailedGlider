#! /usr/bin/env python3
#
# Compute a Hann window and compare to scipy's version
#
# Feb-2022, Pat Welch

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy import signal

def Bias(N:int, window="hann") -> np.array:
    if window == "boxcar": return np.ones(N)
    if window == "hann": 
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(0, N) / N))
    raise NotImplemented

parser = ArgumentParser()
parser.add_argument("--window", type=str, default="boxcar",
        choices=("boxcar", "hann"),
        help="Window to test")
parser.add_argument("n", nargs="?", type=int, help="Window width")
args = parser.parse_args()

if args.n is None:
    args.n = np.power(2, np.arange(6, 14))

for n in args.n:
    w0 = Bias(n, args.window)
    w1 = signal.windows.get_window(args.window, n)
    print("n", n, "max diff", np.abs(w0-w1).max())



