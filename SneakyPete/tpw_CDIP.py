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

import numpy as np
import xarray as xr

def Bias(width:int, window:str="hann") -> np.array:
    return np.ones(width) if window == "boxcar" else (
        0.5*(1 - np.cos(2*np.pi*np.arange(width) / (width - 0)))
    )

def wfft(data:np.array, width:int, window:str="hann") -> list[np.array]:
    bias = Bias(width, window)

    ffts = []
    for i in range(0, data.size-width+1, width//2):
        w = data[i:i+width]
        ffts.append(np.fft.rfft(w*bias))

    return ffts

def wcalcPSD(
        A_FFT_windows:list[np.array], 
        B_FFT_windows:list[np.array],
        fs:float,
        window:str) -> np.array:
    width = A_FFT_windows[0].size
    spectrums = np.complex128(np.zeros(width))
    for i in range(len(A_FFT_windows)):
        A = A_FFT_windows[i]
        B = B_FFT_windows[i]

        spectrum = calcPSD(A, B, fs, window=window)
        spectrums += spectrum
    return spectrums / len(A_FFT_windows)

def calcPSD(xFFT:np.array, yFFT:np.array, fs:float, window:str) -> np.array:
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

def Data(
        filename:str = "./067.20201225_1200.20201225_1600.nc",
        FREQ:bool = True,
        TXYZ:bool = True,
        WAVE:bool = True,
        META:bool = True,
        ) -> dict:

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
