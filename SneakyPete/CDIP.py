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

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from scipy import signal
import scipy.fft as sp_fft
import sys
# from WaveNumber import waveNumber


FREQ = True
TXYZ = True
WAVE = True
META = True
filename = "./067.20201225_1200.20201225_1600.nc"


def calcPSD(xFFT: np.array, yFFT: np.array, fs: float) -> np.array:
    nfft = xFFT.size
    qEven = nfft % 2
    n = (nfft - 2 * qEven) * 2
    psd = (xFFT.conjugate() * yFFT) / (fs * n)
    if qEven:
        psd[1:] *= 2            # Real FFT -> double for non-zero freq
    else:                       # last point unpaired in Nyquist freq
        psd[1:-1] *= 2          # Real FFT -> double for non-zero freq
    return psd


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
    dx2 = np.zeros(x.shape)
    dx2[2:] = np.diff(np.diff(x))
    dx2[0:2] = dx2[2]
    return dx2 * fs * fs


def Data():
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
            "peak-PSD": wave_xr.PeakPSD
        }

        data["wave"]["time-bounds"] = {
            "lower": wave_xr.TimeBounds[:, 0].to_numpy(),
            "upper": wave_xr.TimeBounds[:, 1].to_numpy()
        }

    if FREQ:
        data["freq"] = {
            "bandwidth": wave_xr.Bandwidth,
            "lower-bound": wave_xr.FreqBounds[:, 0],
            "upper-bound": wave_xr.FreqBounds[:, 1],
        }

    return data
