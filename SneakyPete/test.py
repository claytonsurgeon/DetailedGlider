import argparse
from matplotlib.style import use
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
import matplotlib.pyplot as plt
import datetime
from scipy import signal
import scipy.fft as sp_fft
import sys

def user_input():
    # create a writable nc file
    # calculate wanted data
    # store data in xr.dataset
    # store everything into nc file using to_netcdf
    nc4.Dataset('something.nc', 'w', format='NETCDF4') # creates an nc file   
    declination = input("Enter Declination: ")
    ds = xr.Dataset({
        "Declination": float(declination)
    })
    ds.to_netcdf('something.nc', mode='a', group="Meta")
    

def process(fn:str, args:argparse.ArgumentParser) -> None:
    meta = xr.open_dataset(fn,group="Meta")
    #wave = xr.open_dataset(fn,group="Wave")  
    if (args.nc):
        #print(meta.Declination)
        print(meta)

parser = argparse.ArgumentParser()
parser.add_argument("nc", nargs=1, type=str, help="netCDF file to process")
args = parser.parse_args()

def main(): 
    user_input()
    for fn in args.nc:
        process(fn, args)

if __name__ == "__main__":
    main()