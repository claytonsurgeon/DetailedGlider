

# import Data from CDIP
from CDIP import (Data, calcPSD, wcalcPSD, wfft)
import numpy as np
import matplotlib.pyplot as plt


data = Data()
# print(Data())

time_bounds = data["wave"]["time-bounds"]
time = data["time"]

# print(len(time))
# print(len(time_bounds["lower"]))

# arr = np.array([1, 2, 3, 4])

# print(arr)
# print(
#     arr[np.array([True, False, False, True])]
# )
# print(len(data["freq"]["bandwidth"]), len(data["freq"]["lower-bound"]))
# exit(0)


freq_bounds = data["freq"]["bounds"]
freq_lower = freq_bounds["lower"]
freq_upper = freq_bounds["upper"]
freq_midpoints = freq_bounds["joint"].mean(axis=1)


for i in range(len(time_bounds["lower"])):
    # print(i)
    time_lower = time_bounds["lower"][i]
    time_upper = time_bounds["upper"][i]

    # bit mask so as to select only within the bounds of one lower:upper range pair
    select = np.logical_and(
        time >= time_lower,
        time <= time_upper
    )

    # use select to filter
    # time = data["time"][select]
    acc = {
        "x": data["acc"]["x"][select],      # x is northwards
        "y": data["acc"]["y"][select],      # y is eastwards
        "z": data["acc"]["z"][select]       # z is upwards
    }

    FFT = {
        "x": np.fft.rfft(acc["x"], n=acc["z"].size),  # northwards
        "y": np.fft.rfft(acc["y"], n=acc["z"].size),  # eastwards
        "z": np.fft.rfft(acc["z"], n=acc["z"].size),  # upwards
    }

    wFFT = {
        "x": wfft(acc["x"], 2**8),
        "y": wfft(acc["y"], 2**8),
        "z": wfft(acc["z"], 2**8),
    }

    PSD = {
        # imaginary part is zero
        "xx": calcPSD(FFT["x"], FFT["x"], data["frequency"]).real,
        "yy": calcPSD(FFT["y"], FFT["y"], data["frequency"]).real,
        "zz": calcPSD(FFT["z"], FFT["z"], data["frequency"]).real,

        "xy": calcPSD(FFT["x"], FFT["y"], data["frequency"]),
        "xz": calcPSD(FFT["x"], FFT["z"], data["frequency"]),

        "yz": calcPSD(FFT["y"], FFT["z"], data["frequency"]),
    }

    wPSD = {
        "xx": wcalcPSD(wFFT["x"], wFFT["x"], data["frequency"]).real,
        "freq_space": np.fft.rfftfreq(wFFT["z"][0].size*2-1, 1/data["frequency"])
    }

    freq_space = np.fft.rfftfreq(acc["z"].size, 1/data["frequency"])

    freq_select = np.logical_and(
        np.less_equal.outer(freq_lower, freq_space),
        np.greater_equal.outer(freq_upper, freq_space)
    )

    count = freq_select.sum(axis=1)
    # print(count)
    # print(freq_midpoints)

    Band = {
        "xx": (freq_select * PSD["xx"]).sum(axis=1) / count,
        "yy": (freq_select * PSD["yy"]).sum(axis=1) / count,
        "zz": (freq_select * PSD["zz"]).sum(axis=1) / count,

        "xy": (freq_select * PSD["xy"]).sum(axis=1) / count,
        "xz": (freq_select * PSD["xz"]).sum(axis=1) / count,

        "yz": (freq_select * PSD["yz"]).sum(axis=1) / count,
    }

    a0 = Band["zz"] / np.square(np.square(2 * np.pi * freq_midpoints))
    m0 = (a0 * data["freq"]["bandwidth"]).sum()

    wave = data["wave"]
    print(
        "PeakPSD from CDIP",
        float(wave["peak-PSD"][i]),
        "calc",
        a0.max()
    )
    print("Hs from CDIP", float(wave["sig-height"][i]),
          "4*sqrt(z.var0)", 4 * np.sqrt(acc["z"].var()),
          "4*sqrt(m0)", 4 * np.sqrt(m0))

    fig1, [plt_psd_xx, plt_psd_banded_xx,
           plt_w_psd_xx] = plt.subplots(nrows=3, ncols=1)
    plt_psd_xx.plot(freq_space, PSD["xx"])
    plt_psd_xx.set_ylabel("Amplitude, m/s^2")
    plt_psd_xx.set_xlabel("freq (Hz)")
    plt_psd_xx.set_title('PSD')

    plt_psd_banded_xx.plot(freq_midpoints, Band["xx"])
    plt_psd_banded_xx.set_ylabel("Amplitude, m/s^2")
    plt_psd_banded_xx.set_xlabel("freq (Hz)")

    plt_w_psd_xx.plot(wPSD["freq_space"], wPSD["xx"])
    plt_w_psd_xx.set_ylabel("Amplitude, m/s^2")
    plt_w_psd_xx.set_xlabel("freq (Hz)")

    plt.show()


# q = np.logical_and(
#     time >= time_bounds["lower"],
#     time <= time_bounds["upper"]
# ).to_numpy()
