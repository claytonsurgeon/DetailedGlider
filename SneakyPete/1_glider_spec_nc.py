import wave
from CDIPv2 import (Data, Rolling_mean, calcPSD, wcalcPSD, wfft, Plotter, Bias)
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc4
from argparse import ArgumentParser

def process(fn: str, args: ArgumentParser) -> None:

    # windowing perameters
    if(args.hann):
        window_type = "hann"
    else:
        window_type = "boxcar"

    # call master data function to extract all our data from the .nc file
    data = Data(args.nc[0])
    outputs = []
    
    # The lists below collects all the instances of each variable to store them in the nc file
    meta_grp = "Meta"
    wave_grp = "Wave"
    xyz_grp = "XYZ"
    
    hs_arr = []
    ta_arr = []
    tp_arr = []
    wave_energy_ratio_arr = []
    Tz_arr = []
    Dp_arr = []
    peak_psd_arr = []
    Te_arr = []
    dp_true_arr = []
    dp_mag_arr = []

    # time bounds holds an array starting and ending times for each analisis block
    time_bounds = data["time-bounds"]
    time = data["time"]

    # lists of upper, lower, and midpoint frequencies for banding
    freq_bounds = data["freq"]["bounds"]
    freq_lower = freq_bounds["lower"]
    freq_upper = freq_bounds["upper"]
    freq_midpoints = freq_bounds["joint"].mean(axis=1)

    # loop runs through every analisis block, displaying output calculations
    # and graphs at the end of each loop run
    for i in range(len(time_bounds["lower"])):

        time_lower = time_bounds["lower"][i]
        time_upper = time_bounds["upper"][i]
        # print(time_lower, " - ", time_upper)

        # bit mask so as to select only within the bounds of one lower:upper range pair
        select = np.logical_and(
            time >= time_lower,
            time <= time_upper
        )

        # use select to filter to select the acc data corresponding to the current block
        # time = data["time"][select]
       
        averaging_window = 2
        acc = {
            "x": Rolling_mean(data["acc"]["x"][select], averaging_window),      # x is northwards
            "y": Rolling_mean(data["acc"]["y"][select], averaging_window),      # y is eastwards
            "z": Rolling_mean(data["acc"]["z"][select], averaging_window)       # z is upwards

        }

        # preform FFT on block
        FFT = {
            "x": np.fft.rfft(acc["x"], n=acc["z"].size),  # northwards
            "y": np.fft.rfft(acc["y"], n=acc["z"].size),  # eastwards
            "z": np.fft.rfft(acc["z"], n=acc["z"].size),  # upwards
        }

        # preform FFT on block using welch mothod
        wFFT = {
            "x": wfft(acc["x"], 2**8, window_type),
            "y": wfft(acc["y"], 2**8, window_type),
            "z": wfft(acc["z"], 2**8, window_type),
        }

        # Calculate PSD of data from normal FFT
        PSD = {
            # imaginary part is zero
            "xx": calcPSD(FFT["x"], FFT["x"], data["frequency"], "boxcar").real,
            "yy": calcPSD(FFT["y"], FFT["y"], data["frequency"], "boxcar").real,
            "zz": calcPSD(FFT["z"], FFT["z"], data["frequency"], "boxcar").real,

            "xy": calcPSD(FFT["x"], FFT["y"], data["frequency"], "boxcar"),
            "zx": calcPSD(FFT["z"], FFT["x"], data["frequency"], "boxcar"),
            "zy": calcPSD(FFT["z"], FFT["y"], data["frequency"], "boxcar"),
        }

        # calculate PSD on output from welch method FFT
        wPSD = {
            "xx": wcalcPSD(wFFT["x"], wFFT["x"], data["frequency"], window_type).real,
            "yy": wcalcPSD(wFFT["y"], wFFT["y"], data["frequency"], window_type).real,
            "zz": wcalcPSD(wFFT["z"], wFFT["z"], data["frequency"], window_type).real,

            "xy": wcalcPSD(wFFT["x"], wFFT["y"], data["frequency"], window_type),
            "zx": wcalcPSD(wFFT["z"], wFFT["x"], data["frequency"], window_type),
            "zy": wcalcPSD(wFFT["z"], wFFT["y"], data["frequency"], window_type),

            "freq_space": np.fft.rfftfreq(wFFT["z"][0].size*2-1, 1/data["frequency"])
        }

        # frequency space for plotting FFT
        freq_space = np.fft.rfftfreq(acc["z"].size, 1/data["frequency"])

        # bit mask so as to select only within the bounds of one lower:upper range pair
        freq_select = np.logical_and(
            np.less_equal.outer(freq_lower, freq_space),
            np.greater_equal.outer(freq_upper, freq_space)
        )

        count = freq_select.sum(axis=1)

        windowing_method = Bias(len(freq_space), window_type)

        Band = {
            "xx": (freq_select * PSD["xx"] * windowing_method).sum(axis=1) / count,
            "yy": (freq_select * PSD["yy"] * windowing_method).sum(axis=1) / count,
            "zz": (freq_select * PSD["zz"] * windowing_method).sum(axis=1) / count,

            "xy": (freq_select * PSD["xy"] * windowing_method).sum(axis=1) / count,
            "zx": (freq_select * PSD["zx"] * windowing_method).sum(axis=1) / count,

            "zy": (freq_select * PSD["zy"] * windowing_method).sum(axis=1) / count,
        }


        

        print("Processing Block {0}".format(i))

        ##########################################
        # Calculations
        ####################### ##################

        Output = {}
        outputs.append(Output)
        ##########################################
        # Calculations using the welch method
        ##########################################
        def welch(run: bool):            
            if run == False:
                return

            a0 = wPSD["zz"][1:] / np.square(np.square(2 * np.pi * wPSD["freq_space"][1:]))
            m0 = (a0 * wPSD["freq_space"][1]).sum()
            m1 = (a0*wPSD["freq_space"][1:]*wPSD["freq_space"][1]).sum()
            mm1 = (a0/wPSD["freq_space"][1:]*wPSD["freq_space"][1]).sum()
            te = mm1/m0  # mean energy period
            m2 = (a0*np.square(wPSD["freq_space"][1:]) * wPSD["freq_space"][1]).sum()
            tp = 1/wPSD["freq_space"][1:][a0.argmax()]
            denom = np.sqrt(wPSD["zz"] * (wPSD["xx"] + wPSD["yy"]))
            a1 = wPSD["zx"].imag / denom
            b1 = -wPSD["zy"].imag / denom
            denom = wPSD["xx"] + wPSD["yy"]
            dp = np.arctan2(b1[a0.argmax()], a1[a0.argmax()])  # radians
 
            # Appends data from each block into corresponding variable in a list until
            # the end of the loop. 
            hs_arr.append(4 * np.sqrt(m0))
            ta_arr.append(m0/m1)
            tp_arr.append(tp)
            wave_energy_ratio_arr.append(te/tp)
            Tz_arr.append(np.sqrt(m0/m2))
            Dp_arr.append(np.arctan2(b1[a0.argmax()], a1[a0.argmax()]))
            peak_psd_arr.append(a0.max())
            Te_arr.append(te)
            dp_true_arr.append(np.degrees(dp) % 360)
            dp_mag_arr.append(np.degrees(dp+data["declination"]) % 360)

            # Stores the FFT coefficients into a netCDF file. The rest of the data is stored out of the for loop
            # below because we'll then face a size error since it's constantly looping back and putting things in,
            # and I don't think xr.Dataset likes that. It likes it when everything is stored all at once.
            nc4.Dataset('welch.nc', 'w', format='NETCDF4') # creates an nc file   
            df = xr.Dataset({
                "A1": a1,
                "B1": b1,
                "A2": (wPSD["xx"] - wPSD["yy"]) / denom,
                "B2": -2 * wPSD["xy"].real / denom
            })
            df.to_netcdf("welch.nc", mode="a", group=wave_grp) # writes to nc ile
                
            Output["welch"] = {
                "Hs": 4 * np.sqrt(m0),
                "Ta": m0/m1, #average period
                "Tp": tp,  # peak wave period
                "wave_energy_ratio": te/tp,
                "Tz": np.sqrt(m0/m2),
                "Dp": np.arctan2(b1[a0.argmax()], a1[a0.argmax()]),
                "PeakPSD": a0.max(),
                "te": te, # mean energy period
                "dp_true": np.degrees(dp) % 360,
                "dp_mag": np.degrees(dp+data["declination"]) % 360,
                "a1": a1,
                "b1": b1,
                "a2": (wPSD["xx"] - wPSD["yy"]) / denom,
                "b2": -2 * wPSD["xy"].real / denom,
            }
            print("Calculated Data using Welch method \"{0}\" window: ".format(window_type))
            for j in Output["welch"]:
                if np.isscalar(Output["welch"][j]):
                    print(j, "=", Output["welch"][j])
        welch(args.welch)


        ##########################################
        # Calculations using the banded method
        ##########################################

        def banded(run: bool):
            # Preform Baniding on the PSD. Averages the data withen each bin.  
            if run == False:
                return
            a0 = Band["zz"] / np.square(np.square(2 * np.pi * freq_midpoints))
            m0 = (a0 * data["freq"]["bandwidth"]).sum()
            m1 = (a0*freq_midpoints*data["freq"]["bandwidth"]).sum()
            mm1 = (a0/freq_midpoints*data["freq"]["bandwidth"]).sum()
            te = mm1/m0  # mean energy period
            m2 = (a0*np.square(freq_midpoints) * data["freq"]["bandwidth"]).sum()
            tp = 1/freq_midpoints[a0.argmax()]
            denom = np.sqrt(Band["zz"] * (Band["xx"] + Band["yy"]))
            a1 = Band["zx"].imag / denom
            b1 = -Band["zy"].imag / denom
            denom = Band["xx"] + Band["yy"]
            dp = np.arctan2(b1[a0.argmax()], a1[a0.argmax()])  # radians

            # Appends data from each block into corresponding variable in a list until
            # the end of the loop.  
            hs_arr.append(4 * np.sqrt(m0))
            ta_arr.append(m0/m1)
            tp_arr.append(tp)
            wave_energy_ratio_arr.append(te/tp)
            Tz_arr.append(np.sqrt(m0/m2))
            Dp_arr.append(np.arctan2(b1[a0.argmax()], a1[a0.argmax()]))
            peak_psd_arr.append(a0.max())
            Te_arr.append(te)
            dp_true_arr.append(np.degrees(dp) % 360)
            dp_mag_arr.append(np.degrees(dp+data["declination"]) % 360)

            # Stores FFT coefficients 
            nc4.Dataset('bandings.nc', 'w', format='NETCDF4') # creates an nc file
            df = xr.Dataset({
                "A1": a1,
                "B1": b1,
                "A2": (Band["xx"] - Band["yy"]) / denom,
                "B2": -2 * Band["xy"].real / denom
            })
            df.to_netcdf("bandings.nc", mode="a", group=wave_grp) # writes to nc file

            Output["banded"] = {
                "Hs": 4 * np.sqrt(m0),
                "Ta": m0/m1,
                "Tp": tp,  # peak wave period
                "wave_energy_ratio": te/tp,
                "Tz": np.sqrt(m0/m2),
                "Dp": np.arctan2(b1[a0.argmax()], a1[a0.argmax()]),
                "PeakPSD": a0.max(),
                "te": te,
                "dp_true": np.degrees(dp) % 360,
                "dp_mag": np.degrees(dp+data["declination"]) % 360,
                "a1": a1,
                "b1": b1,
                "a2": (Band["xx"] - Band["yy"]) / denom,
                "b2": -2 * Band["xy"].real / denom,
            }
            print("Calculated Data using Banding and \"{0}\" window: ".format(window_type))
            for j in Output["banded"]:
                if np.isscalar(Output["banded"][j]):
                    print(j, "=", Output["banded"][j])
        banded(args.banding)

        print("\n\nCDIP Data: ")
        for j in data["wave"]:
            if np.isscalar(data["wave"][j][i]):
                print(j, "=", data["wave"][j][i])

    

        ##########################################
        # plotting
        ##########################################

        if(args.raw):
            figure = [
                ["X Acc", "", "m/s^2", data["time"][select], acc["x"]],
                ["Y Acc", "", "m/s^2", data["time"][select], acc["y"]],
                ["Z Acc", "Time (s)", "m/s^2", data["time"][select], acc["z"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.norm and args.graph):
            figure = [
                ["X PSD", "", "", freq_space, PSD["xx"]],
                ["Y PSD", "", "", freq_space, PSD["yy"]],
                ["Z PSD", "freq (Hz)", "", freq_space, PSD["zz"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.banding and args.graph):
            figure = [
                ["X Banded PSD", "", "", freq_midpoints, Band["xx"]],
                ["Y Banded PSD", "", "", freq_midpoints, Band["yy"]],
                ["Z Banded PSD", "freq (Hz)", "", freq_midpoints, Band["zz"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.welch and args.graph):
            figure = [
                ["X Windowed PSD", "", "", wPSD["freq_space"], wPSD["xx"]],
                ["Y Windowed PSD", "", "", wPSD["freq_space"], wPSD["yy"]],
                ["Z Windowed PSD", "freq (Hz)", "",
                 wPSD["freq_space"], wPSD["zz"]]
            ]
            fig, axs = plt.subplots(nrows=3, ncols=1)
            Plotter(fig, axs, figure)

        if(args.ds):
            if args.banding:
                figure = [
                    ["Directional Spectra with banding", "", "A1",
                        freq_midpoints, [Output["banded"]["a1"], data["wave"]["a1"][i]]],
                    ["", "", "B1", freq_midpoints, [Output["banded"]["b1"], data["wave"]["b1"][i]]],
                    ["", "", "A2", freq_midpoints, [Output["banded"]["a2"], data["wave"]["a2"][i]]],
                    ["", "freq (Hz)", "B2", freq_midpoints, [Output["banded"]["b2"], data["wave"]["b2"][i]]]
                ]
                fig, axs = plt.subplots(nrows=4, ncols=1)
                Plotter(fig, axs, figure)
            elif(args.welch):
                figure = [
                    ["Directional Spectra with welch", "", "A1",
                        [wPSD["freq_space"], freq_midpoints], [Output["welch"]["a1"], data["wave"]["a1"][i]]],
                    ["", "", "B1", [wPSD["freq_space"], freq_midpoints], [Output["welch"]["b1"], data["wave"]["b1"][i]]],
                    ["", "", "A2", [wPSD["freq_space"], freq_midpoints], [Output["welch"]["a2"], data["wave"]["a2"][i]]],
                    ["", "freq (Hz)", "B2", [wPSD["freq_space"], freq_midpoints], [Output["welch"]["b2"], data["wave"]["b2"][i]]]
                ]
                fig, axs = plt.subplots(nrows=4, ncols=1)
                Plotter(fig, axs, figure)
            else:
                print("Error: please enter calculation option (Welch or Banding)")
                exit(0)
            
         
                
        if(args.welch or args.banding or args.raw or args.ds or args.norm):
            plt.show()

        print("\n--------------------------\n")

        # exit(0)  # comment out if you want to proccess all the blocks of data
    
    ##########################################
    # Storing Calculations into netCDF file
    ##########################################

    # Out of the for loop so it has all the instances from the blocks in their corresponding list
    # Turns lists into np arrays, then stored in netCDF file. 
    # Coefficients (a1,b2,a1,b2) for the FFT are not here but above because it needs the calculations, which is out of scope
    # I can append the coefficients into a list and turn it into an np array like the rest, but it'll throw 
    # an error, and I think it's because a1,b1,a2,b2 is already in a list unlike the others, and xr.Dataset
    # doesn't like a list that's appended with a bunch of other lists.
    if (args.welch):
        Hs = np.asarray(hs_arr)
        Ta = np.asarray(ta_arr)
        Tp = np.asarray(tp_arr)
        wave_energy = np.asarray(wave_energy_ratio_arr)
        Tz = np.asarray(Tz_arr) 
        Dp = np.asarray(Dp_arr)
        peak_psd = np.asarray(peak_psd_arr)
        Te = np.asarray(Te_arr)
        dp_true = np.asarray(dp_true_arr)
        dp_mag = np.asarray(dp_mag_arr) 
        df = xr.Dataset({
            "Hs": Hs,
            "Ta": Ta,
            "Tp": Tp,
            "Wave_Energy_Ration": wave_energy,
            "Tz": Tz,
            "Dp": Dp,
            "Peak Psd": peak_psd,
            "Te": Te,
            "Dp_true": dp_true,
            "Dp_mag": dp_mag,
        })
        df.to_netcdf("welch.nc", mode="a", group=wave_grp)

    if (args.banding):
        Hs = np.asarray(hs_arr)
        Ta = np.asarray(ta_arr)
        Tp = np.asarray(tp_arr)
        wave_energy = np.asarray(wave_energy_ratio_arr)
        Tz = np.asarray(Tz_arr) 
        Dp = np.asarray(Dp_arr)
        peak_psd = np.asarray(peak_psd_arr)
        Te = np.asarray(Te_arr)
        dp_true = np.asarray(dp_true_arr)
        dp_mag = np.asarray(dp_mag_arr) 
        df = xr.Dataset({
            "Hs": Hs,
            "Ta": Ta,
            "Tp": Tp,
            "Wave_Energy_Ration": wave_energy,
            "Tz": Tz,
            "Dp": Dp,
            "Peak Psd": peak_psd,
            "Te": Te,
            "Dp_true": dp_true,
            "Dp_mag": dp_mag,
        })
        df.to_netcdf("bandings.nc", mode="a", group=wave_grp)   


def main():

    #######################################
    # command line stuff
    #######################################
    parser = ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    
    # calculation options
    grp.add_argument("--welch", action="store_true", help="Welch Method")
    grp.add_argument("--banding", action="store_true", help="Banding Method")
    
    # optional args
    parser.add_argument("--hann", action="store_true", help="to choose hann windowing method")
    parser.add_argument("--boxcar", action="store_true", help="to choose boxcar windowing method")
    parser.add_argument("--norm", action="store_true", help="Normal FFT PSD")
    parser.add_argument("--ds", action="store_true", help="Directional Spectrum coefficients")
    parser.add_argument("--graph", action="store_true", help="Turns graphs on")
    parser.add_argument("--raw", action="store_true", help="Raw acceleration data")

    # required
    parser.add_argument("nc", nargs="+", type=str, help="netCDF file to process")  # typed after commands

    args = parser.parse_args()

    for fn in args.nc:
        process(fn, args)



if __name__ == "__main__":
    main()