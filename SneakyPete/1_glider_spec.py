

# import Data from CDIP
from CDIP import Data
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


# exit(0)

for i in range(len(time_bounds["lower"])):
    # print(i)
    lower = time_bounds["lower"][i]
    upper = time_bounds["upper"][i]

    # bit mask so as to select only within the bounds of one lower:upper range pair
    select = np.logical_and(
        time >= lower,
        time <= upper
    )

    # use select to filter
    # time = data["time"][select]
    accx = data["acc"]["x"][select]     # x is northwards
    accy = data["acc"]["y"][select]     # y is eastwards
    accz = data["acc"]["z"][select]     # z is upwards

    xFFT = np.fft.rfft(accx, n=accz.size)  # northwards
    yFFT = np.fft.rfft(accy, n=accz.size)  # eastwards
    zFFT = np.fft.rfft(accz, n=accz.size)  # upwards

    freq_space = np.fft.rfftfreq(accz.size, 1/data["frequency"])

    plt.plot(freq_space, xFFT)
    plt.ylabel("Amplitude, m/s^2")
    plt.xlabel("freq (Hz)")
    plt.title('freq Domain')
    plt.show()


# q = np.logical_and(
#     time >= time_bounds["lower"],
#     time <= time_bounds["upper"]
# ).to_numpy()
