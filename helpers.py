import math
import numpy as np
import csv

# temporary
class Calculation:
    duration = 20*60 # 20 minutes
    samples = 2**12 # 4096 samples
    fs = duration / (samples - 1) # frequency samples
    time = np.linspace(0, duration, samples, endpoint=False) # time
    wave = 10*np.sin(2*np.pi*time * (1/1)) + 2*np.cos(2*np.pi* time * (1/30)) # mock wave
    freq_space = np.fft.rfftfreq(n=samples, d=fs) 
    num_of_windows = samples // 2**8
    M = samples // num_of_windows
    hann_window = 0.5*(1 - np.cos(2*np.pi*np.array(range(0, M)) / (M - 0)))

class Direction():
    t = []
    x = []
    y = []
    z = []

def parse_csv():
    t = []
    x = []
    y = []
    z = []
    direction = []
    with open('Output.csv', newline='') as csvfile:
        #https://docs.python.org/3/library/csv.html
        #https://evanhahn.com/python-skip-header-csv-reader/
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #next(csvfile)
        for row in spamreader:
            # According to the csv file
            t.append((float(row[0])))
            x.append((float(row[1])))
            y.append((float(row[2])))
            z.append((float(row[3])))

    return [t, x, y, z]

        #print(Directions.x)
        #print(Directions.y)
        #print(Directions.z)

######################
#error in the mean = Standard Deviation(data) / sqrt(number samples)
####################
def calcPSD(xAxis, yAxis):
    windows = []
    N = Calculation.M//2

    for i in range(0, Calculation.samples-Calculation.M+1, N):
        next_window = Calculation.wave[i:i+Calculation.M] 
        windows.append(next_window)

    spectrums = np.zeros(N+1)
    denominator = Calculation.hann_window.sum() * Calculation.fs

    #for window in windows:
    #    spectrum1 = (xAxis.conjugate() * yAxis).real / denominator
    #    spectrum1[1:-1] *= 2
    #    spectrums += spectrum1
    
    #PSD = spectrums/len(windows) * Calculation.num_of_windows
    #return PSD

    spectrum1 = (xAxis.conjugate() * yAxis).real / denominator
    spectrum1[1:-1] *= 2

    return spectrum1


def normalize(data, wave_data, output, sampling_freq, samples):
    # get the magnitude of the spectrum then normalize by number of 'buckets' in the spectrum
    spectrum = []
    if(output == 'power'):
        spectrum = (data.conjugate() * data).real / (wave_data.size * sampling_freq)
        spectrum[1:-1] *= 2
    elif(output == 'amplitude'):
        # spectrum = np.sqrt(data.conjugate()*data) / (samples/2)
        spectrum = np.abs(np.fft.rfft(wave_data)) / (samples/2)
    else:
        print("error: invalid output")
        exit(0)
    return spectrum


"""
preforms FFT using windowing method. "hann" for hann windowing and "boxcar" for square windowing
"""
def windowfft(type, samples, M, num_of_windows, wave, sampling_freq):
    hann_window = []
    window = []
    if(type == 'boxcar'):
        hann_window = np.ones(M)
        pass
    elif(type == 'hann'):
        hann_window = 0.5*(1 - np.cos(2*np.pi*np.array(range(0, M)) / (M - 0)))
        pass
    else:
        print("error: invalid type")
        exit(0)
    
    windows = []
    N = M//2

    for i in range(0, samples-M+1, N):
        next_window = wave[i:i+M]
        windows.append(next_window)

    spectrums = np.zeros(N+1)

    denominator = hann_window.sum() * sampling_freq
    for window in windows:        
        A = np.fft.rfft(window*hann_window)

        spectrum1 = (A.conjugate() * A).real / denominator
        spectrum1[1:-1] *= 2
        spectrums += spectrum1

    final_thing = spectrums/len(windows) * num_of_windows
    
    return final_thing


#################
# 
#################
def getSWH(wave):
    
    # freqBounds = wave.FreqBounds[:,:].to_numpy()
    # fMid = freqBounds.mean(axis=1)
    # a0 = zzBand / np.square(np.square(2 * np.pi * fMid))
    

    # sd = np.std(spectrum)
    
    return 0





#takes two arrays representing the x (frequency) and y (PSD) coordinates for an acceleration spectrum density graph. 
#returns the PSD array cleaned up
def clean_up(data):
    newArr = list(data)

    #getting the AS(12deltaf) and AS(24deltaf)
    ASF12 = np.argmin(np.abs(np.array([i[0] for i in newArr])-0.01))
    ASF24 = np.argmin(np.abs(np.array([i[0] for i in newArr])-0.02))

    #print out AS(12deltaf) and AS(24deltaf) information for testing
    print("ASF12: \n\tFrequency = ", (newArr[ASF12][0]), "\n\tPSD = ", (newArr[ASF12][1]), "\n", "ASF24: \n\tFrequency =", (newArr[ASF24][0]), "\n\tPSD = ", (newArr[ASF24][1]))

    #calculating GU
    GU = ((newArr[ASF12][1]) + (newArr[ASF24][1]))/2.0
    NC = 0
    #loop that runs through the points and updates the PSD value based on the Data-dependent noise function. 
    for i in range(len(data)): 
        
        #checking bounds. If the frequency is below .15Hz, then apply the data-dependent noise function
        if(data[i][0]<0.15):
            
            #calculating the data dependent noise function for this frequency and psd value. 
            NC = newArr[i][1] = 13*GU*(0.15-data[i][0])
            
            #set PSD to zero if frequency is above .05, and the psd value - NC is less than zero,  
            #or if the frequency is below 0.05.  
            if((data[i][0] > .05 and newArr[i][1] - NC <= 0) or data[i][0] <= 0.05):
                newArr[i][1] = 0

            #if frequency is above 0.05 and PSD - NC is greater than zero, apply data-dependent noise function. 
            elif(data[i][0] > 0.05 and newArr[i][1] - NC > 0):
                newArr[i][1] = data[i][1] - NC

    return newArr



