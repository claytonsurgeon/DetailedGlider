#! /usr/bin/env python3
#
# Load the CDIP data that was fetched by extract.py
# and do some processing on it.
#
# URLs:
# https://docs.google.com/document/d/1Uz_xIAVD2M6WeqQQ_x7ycoM3iKENO38S4Bmn6SasHtY/pub
#
# Hs:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/compendium.html
#
# Hs Boxplot:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/annualHs_plot.html
#
# Sea Surface Temperature:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/temperature.html
#
# Polar Spectrum:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/polar.html
#
# Wave Direction and Energy Density by frequency bins:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html
#
# XYZ displacements:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/dw_timeseries.html
#
# The Datawell documentation is very useful:
# https://www.datawell.nl/Portals/0/Documents/Manuals/datawell_manual_libdatawell.pdf
#
# Dec-2021, Pat Welch

# import argparse
import numpy as np
# import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import datetime
# from scipy import signal
# import scipy.fft as sp_fft
# import sys
# from WaveNumber import waveNumber


FREQ = True
TXYZ = True
WAVE = True
META = True
filename = "./067.20201225_1200.20201225_1600.nc"

# old
# def Bias(width, type="hann"):
#     """returns a either a boxcar, or hann window"""
#     return np.ones(width) if type == "boxcar" else (
#         0.5*(1 - np.cos(2*np.pi*np.array(range(0, width)) / (width - 0)))
#     )

# new
def Bias(width:int, window:str="hann") -> np.array:
    """returns a either a boxcar, or hann window"""
    return np.ones(width) if window == "boxcar" else (
        0.5*(1 - np.cos(2*np.pi*np.arange(width) / (width - 0)))
    )


def wfft(data, width, type="hann"):
    """Splits the acceleration data into widows, 
    preforms FFTs on them returning a list of all the windows
    """
    bias = Bias(width, type)
    windows = []
    for i in range(0, data.size-width+1, width//2):
        window = data[i:i+width]
        windows.append(np.fft.rfft(window*bias))

    return windows


def wcalcPSD(A_FFT_windows: np.array, B_FFT_windows: np.array, frequency: float) -> np.array:
    """calculates the PSD of the FFT output preformed with the windowing method.
    After calculateing the PSD of each window, the resulting lists are averaged together"""
    width = A_FFT_windows[0].size
    spectrums = np.complex128(np.zeros(width))
    
    for i in range(len(A_FFT_windows)):
        A = A_FFT_windows[i]
        B = B_FFT_windows[i]
        spectrum = calcPSD(A, B, frequency)
        spectrums += spectrum

    return spectrums / len(A_FFT_windows)


##################################
# maybe ask about how this works? We normalize for 
# PSD similar to how we did it initally, but im 
# not sure what we are doing with evens and odds. 
##################################

# old
# def calcPSD(xFFT: np.array, yFFT: np.array, fs: float) -> np.array:
#     "calculates the PSD on an output of a FFT"
#     nfft = xFFT.size
#     qEven = nfft % 2
#     n = (nfft - 2 * qEven) * 2
#     psd = (xFFT.conjugate() * yFFT) / (fs * n)
#     if qEven:
#         psd[1:] *= 2            # Real FFT -> double for non-zero freq
#     else:                       # last point unpaired in Nyquist freq
#         psd[1:-1] *= 2          # Real FFT -> double for non-zero freq
#     return psd


# new
def calcPSD(xFFT:np.array, yFFT:np.array, fs:float, window:str) -> np.array:
    "calculates the PSD on an output of a FFT"
    nfft = xFFT.size
    qOdd = nfft % 2
    n = (nfft - qOdd) * 2 # Number of data points input to FFT
    w = Bias(n, window) # Get the window used
    wSum = (w * w).sum()
    psd = (xFFT.conjugate() * yFFT) / (fs * wSum)
    if not qOdd:       # Even number of FFT bins
        psd[1:] *= 2   # Real FFT -> double for non-zero freq
    else:              # last point unpaired in Nyquist freq
        psd[1:-1] *= 2 # Real FFT -> double for non-zero freq
    return psd


##################################
# Im not sure what this is for. We dont use this anywhere
##################################
def zeroCrossingAverage(z: np.array, fs: float) -> float:
    q = z[0:-1] * z[1:] < 0         # opposite times between successive zs
    iLHS = np.flatnonzero(q)        # Indices of LHS
    iRHS = iLHS + 1
    zLHS = z[iLHS]
    zRHS = z[iRHS]
    # Fraction of zRHS,zLHS interval to zero from zLHS
    zFrac = -zLHS / (zRHS - zLHS)
    tZero = zFrac / fs              # Seconds from iLHS to the zero crossing point
    dt = np.diff(iLHS) / fs         # interval between iLHS
    dt += zFrac[1:] / fs            # Add in time from RHS to zero crossing
    dt -= zFrac[0:-1] / fs          # Take off tiem from LHS to zero crossing
    return 2 * dt.mean()            # 2 times the half wave zero crossing average time


def calcAcceleration(x: np.array, fs: float) -> np.array:
    """converts displacement data to acceleration.
    We need acceleration data because that is
    what we will record from the STM"""
    dx2 = np.zeros(x.shape)
    dx2[2:] = np.diff(np.diff(x))
    dx2[0:2] = dx2[2]
    return dx2 * fs * fs


def Data():
    """Master data reading function. Reads the .nc file from CDIP.
    The data is stored in dictionary (data), which contains many dictionaries 
    to hold information. Examples include: acceleration data, frequency bounds, 
    expected values calculated by CDIP, etc."""
    meta_xr = xr.open_dataset(filename, group="Meta")  # For water depth
    wave_xr = xr.open_dataset(filename, group="Wave")
    xyz_xr = xr.open_dataset(filename, group="XYZ")

    depth = float(meta_xr.WaterDepth)
    declination = float(meta_xr.Declination)
    frequency = float(xyz_xr.SampleRate)

    data = {}
    if META:
        data["frequency"] = frequency
        data["latitude"] = float(meta_xr.DeployLatitude)
        data["longitude"] = float(meta_xr.DeployLongitude)
        data["depth"] = depth
        data["declination"] = declination

    if TXYZ:
        data["time"] = xyz_xr.t.to_numpy()
        data["dis"] = {
            "t": data["time"],
            "x": xyz_xr.x.to_numpy(),
            "y": xyz_xr.y.to_numpy(),
            "z": xyz_xr.z.to_numpy()
        }
        data["acc"] = {
            "t": data["time"],
            "x": calcAcceleration(xyz_xr.x.to_numpy(), frequency),
            "y": calcAcceleration(xyz_xr.y.to_numpy(), frequency),
            "z": calcAcceleration(xyz_xr.z.to_numpy(), frequency)
        }

    if WAVE:
        data["wave"] = {
            "sig-height": wave_xr.Hs,
            "avg-period": wave_xr.Ta,
            "peak-period": wave_xr.Tp,
            "mean-zero-upcross-period": wave_xr.Tz,
            "peak-direction": wave_xr.Dp,
            "peak-PSD": wave_xr.PeakPSD,
            "a1": wave_xr.A1,
            "b1": wave_xr.B1,
            "a2": wave_xr.A2,
            "b2": wave_xr.B2,
        }

        data["wave"]["time-bounds"] = {
            "lower": wave_xr.TimeBounds[:, 0].to_numpy(),
            "upper": wave_xr.TimeBounds[:, 1].to_numpy()
        }

    if FREQ:
        data["freq"] = {
            "bandwidth": wave_xr.Bandwidth.to_numpy(),
        }
        data["freq"]["bounds"] = {
            "lower": wave_xr.FreqBounds[:, 0].to_numpy(),
            "upper": wave_xr.FreqBounds[:, 1].to_numpy(),
            "joint": wave_xr.FreqBounds[:, :].to_numpy()
        }

    return data


