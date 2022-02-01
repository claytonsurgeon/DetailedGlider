#! /usr/bin/env python3

# import Data from tpw_CDIP
from tpw_CDIP import (Data, calcPSD, wcalcPSD, wfft)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

window = "boxcar"
nPerSeg = 2**8 # Welch segment length

data = Data()
# print(data)

time_bounds = data["wave"]["time-bounds"]
time = data["time"]

# print("time n", len(time), "min", time.min(), "max", time.max())
# print("time bounds n", len(time_bounds["lower"]))
# sys.exit(1)

freq_bounds = data["freq"]["bounds"]
freq_lower = freq_bounds["lower"]
freq_upper = freq_bounds["upper"]

fs = data["frequency"] #  sampling frequency in Hz
print("fs", fs, "Window", window)

def mkBands(freq_lower:np.array, freq_upper:np.array, psd:dict) -> dict:
    freq_select = np.logical_and(
            np.less_equal.outer(freq_lower, psd["f"]),
            np.greater_equal.outer(freq_upper, psd["f"])
            )

    count = freq_select.sum(axis=1)

    return {
            "xx": (freq_select * psd["xx"]).sum(axis=1) / count,
            "xy": (freq_select * psd["xy"]).sum(axis=1) / count,
            "xz": (freq_select * psd["xz"]).sum(axis=1) / count,

            "yx": (freq_select * psd["yx"]).sum(axis=1) / count,
            "yy": (freq_select * psd["yy"]).sum(axis=1) / count,
            "yz": (freq_select * psd["yz"]).sum(axis=1) / count,

            "zx": (freq_select * psd["zx"]).sum(axis=1) / count,
            "zy": (freq_select * psd["zy"]).sum(axis=1) / count,
            "zz": (freq_select * psd["zz"]).sum(axis=1) / count,

            "f": (freq_lower + freq_upper) / 2
            }

for i in range(len(time_bounds["lower"])):
    time_lower = time_bounds["lower"][i]
    time_upper = time_bounds["upper"][i]

    # bit mask so as to select only within the bounds of one lower:upper range pair
    select = np.logical_and(
        time >= time_lower,
        time <= time_upper
    )

    print("i", i, "stime", time_lower, "etime", time_upper, "n", select.sum())

    # use select to filter
    # time = data["time"][select]
    acc = {
        "x": data["acc"]["x"][select],      # x is northwards
        "y": data["acc"]["y"][select],      # y is eastwards
        "z": data["acc"]["z"][select]       # z is upwards
    }

    FFT = {
        "x": np.fft.rfft(acc["x"]),  # northwards
        "y": np.fft.rfft(acc["y"]),  # eastwards
        "z": np.fft.rfft(acc["z"]),  # upwards
        "f": np.fft.rfftfreq(acc["x"].size, 1/fs)
    }

    wFFT = {
        "x": wfft(acc["x"], nPerSeg, window),
        "y": wfft(acc["y"], nPerSeg, window),
        "z": wfft(acc["z"], nPerSeg, window),
        "f": np.fft.rfftfreq(nPerSeg, 1/fs)
    }

    PSD = {
        # imaginary part is zero
        "xx": calcPSD(FFT["x"], FFT["x"], fs, window).real,
        "yy": calcPSD(FFT["y"], FFT["y"], fs, window).real,
        "zz": calcPSD(FFT["z"], FFT["z"], fs, window).real,

        "xy": calcPSD(FFT["x"], FFT["y"], fs, window),
        "xz": calcPSD(FFT["x"], FFT["z"], fs, window),

        "yz": calcPSD(FFT["y"], FFT["z"], fs, window),
        "yx": calcPSD(FFT["y"], FFT["x"], fs, window), ##

        "zx": calcPSD(FFT["z"], FFT["x"], fs, window), ##
        "zy": calcPSD(FFT["z"], FFT["y"], fs, window), ##

        "f": FFT["f"]
    }

    wPSD = {
        "xx": wcalcPSD(wFFT["x"], wFFT["x"], fs, window).real,
        "yy": wcalcPSD(wFFT["y"], wFFT["y"], fs, window).real,
        "zz": wcalcPSD(wFFT["z"], wFFT["z"], fs, window).real,

        "xy": wcalcPSD(wFFT["x"], wFFT["y"], fs, window),
        "xz": wcalcPSD(wFFT["x"], wFFT["z"], fs, window),

        "yz": wcalcPSD(wFFT["y"], wFFT["z"], fs, window),
        "yx": wcalcPSD(wFFT["y"], wFFT["x"], fs, window), ##

        "zx": wcalcPSD(wFFT["z"], wFFT["x"], fs, window), ##
        "zy": wcalcPSD(wFFT["z"], wFFT["y"], fs, window), ##

        "f": wFFT["f"]
    }

    Band  = mkBands(freq_lower, freq_upper, PSD)
    wBand = mkBands(freq_lower, freq_upper, wPSD)

    ##########################################
    # sig wave height
    ##########################################

    a0  = Band["zz"] / np.square(np.square(2 * np.pi * Band["f"]))
    m0  = (a0  * data["freq"]["bandwidth"]).sum()

    wave = data["wave"]

    # print("Hs from CDIP", float(wave["sig-height"][i]),
    #       "4*sqrt(z.var0)", 4 * np.sqrt(acc["z"].var()),
    #       "4*sqrt(m0)", 4 * np.sqrt(m0))
    print("4*sqrt(z.var0) =", 4 * np.sqrt(acc["z"].var()),
          "\n4*sqrt(m0) = ", 4 * np.sqrt(m0),
          "\nHs from CDIP = ", float(wave["sig-height"][i]),
          )

    ##########################################
    # peak psd
    ##########################################
    print(
        "PeakPSD from CDIP",
        float(wave["peak-PSD"][i]),
        "calc",
        a0.max()
    )
    ##########################################
    # a1, b1, a2, b2
    ##########################################
    denom = np.sqrt(Band["zz"] * (Band["xx"] + Band["yy"]))

    a1 =  Band["zx"].imag / denom
    b1 = -Band["zy"].imag / denom

    denom = Band["xx"] + Band["yy"]

    a2 = (Band["xx"] - Band["yy"]) / denom
    b2 = -2 * Band["xy"].real / denom

    # print(
    #     "a1 = ", a1, "\n expected = ", data["wave"]["a1"], "\n"
    #     "b1 = ", b1, "\n expected = ", data["wave"]["b1"], "\n"
    #     "a2 = ", a2, "\n expected = ", data["wave"]["a2"], "\n"
    #     "b2 = ", b2, "\n expected = ", data["wave"]["b2"], "\n"

    # )

    ##########################################
    # dominant period
    ##########################################
    print("Tp from CDIP", float(data["wave"]["peak-period"][i]),
          "calc", 1/Band["f"][a0.argmax()],
          "not banded", 1/PSD["f"][PSD["zz"].argmax()])

    ##########################################
    # plotting
    ##########################################
    fig1, [plt_psd_xx, plt_psd_banded_xx,
           plt_w_psd_xx] = plt.subplots(nrows=3, ncols=1)
    plt_psd_xx.plot(PSD["f"], PSD["xx"])
    plt_psd_xx.set_ylabel("Amplitude, m/s^2")
    plt_psd_xx.set_xlabel("freq (Hz)")
    plt_psd_xx.set_title('PSD')

    plt_psd_banded_xx.plot(Band["f"], Band["xx"])
    plt_psd_banded_xx.set_ylabel("Amplitude, m/s^2")
    plt_psd_banded_xx.set_xlabel("freq (Hz)")

    plt_w_psd_xx.plot(wPSD["f"], wPSD["xx"])
    plt_w_psd_xx.set_ylabel("Amplitude, m/s^2")
    plt_w_psd_xx.set_xlabel("freq (Hz)")

    (fig, ax) = plt.subplots(4)
    ax[0].plot(Band["f"], wave["a1"][i,:], '-')
    ax[0].plot(Band["f"], a1, '-')
    ax[1].plot(Band["f"], wave["b1"][i,:], '-')
    ax[1].plot(Band["f"], b1, '-')
    ax[2].plot(Band["f"], wave["a2"][i,:], '-')
    ax[2].plot(Band["f"], a2, '-')
    ax[3].plot(Band["f"], wave["b2"][i,:], '-')
    ax[3].plot(Band["f"], b2, '-')
    
    plt.show()
    exit(0)
