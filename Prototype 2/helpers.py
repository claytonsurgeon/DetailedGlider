import numpy as np
import csv
import datetime

#Temp class for Pat's data
class Temp():
    temp = []
    ts = []
    x = []
    y = []
    z = []
    fLower = []
    fUpper = []
    bandwidth = []

class Wave():
    tLower = []
    tUpper = []
    Hs = []
    Ta = []
    Tp = []
    Tz = [] 
    Dp = []
    PeakPSD = []
    
class Direction():
    t = []
    x = []
    y = []
    z = []    

class Block():
    t = []
    x = []
    y = []
    z = []



def parse_wave_data(file):
    wave = Wave()
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        for row in spamreader:
            # According to the csv file
            wave.tLower.append(row[0])
            wave.tUpper.append(row[1])
            wave.Hs.append(float(row[2]))
            wave.Ta.append(float(row[3]))
            wave.Tp.append(float(row[2]))
            wave.Tz.append(float(row[3]))
            wave.Dp.append(float(row[2]))
            wave.PeakPSD.append(float(row[3]))
    return wave        

    

# Parses data for frequency sample
def parse_fs(file):
    fs = []
    with open(file, newline='') as csvfile:
        #https://docs.python.org/3/library/csv.html
        #https://evanhahn.com/python-skip-header-csv-reader/
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        for row in spamreader:
            # According to the csv file
            fs = float(row[0])
    return fs


# Parses data for the time values 
def parse_ts(file):
    data = Temp()
    with open(file, newline='') as csvfile:
        #https://docs.python.org/3/library/csv.html
        #https://evanhahn.com/python-skip-header-csv-reader/
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        for row in spamreader:
            # According to the csv file
            data.temp.append(row[0])
            data.x.append(float(row[1]))
            data.y.append(float(row[2]))
            data.z.append(float(row[3]))

    # Time conversion
    start = datetime.datetime.strptime(str(data.temp[0])[:-3], '%Y-%m-%d %H:%M:%S.%f')
    stringvar = ""
    with open("t_values.csv", "w") as csv_file:
        for i in range(len(data.temp)):
            date = datetime.datetime.strptime(str(data.temp[i])[:-3], '%Y-%m-%d %H:%M:%S.%f')
            stringvar += str(date.timestamp()-start.timestamp()) + ', ' + '\n'
        csv_file.write(stringvar)
    
    with open("t_values.csv", newline='') as csvfile:
        #https://docs.python.org/3/library/csv.html
        #https://evanhahn.com/python-skip-header-csv-reader/
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # According to the csv file
            data.ts.append(float(row[0]))
   
    '''
    The reason for this mess is because when I'm trying to use the raw value of stringvar on line 51,
    and append it to data.ts, it would randomly print itself lwhen calling the function without actually printing it. 
    A way to solve this issue I found was to just store it in a csv file and parse through it for the time values,
    then store it in data.ts
    '''
    return data


# Parses data for the lower and upper frequency bounds
def parse_frequency(file):
    data = Temp()
    with open(file, newline='') as csvfile:
        #https://docs.python.org/3/library/csv.html
        #https://evanhahn.com/python-skip-header-csv-reader/
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        for row in spamreader:
            # According to the csv file
            data.bandwidth.append((float(row[0])))
            data.fLower.append((float(row[1])))
            data.fUpper.append((float(row[2])))
        

    return data


def parse_csv(file):
    data = Direction()
    with open(file, newline='') as csvfile:
        #https://docs.python.org/3/library/csv.html
        #https://evanhahn.com/python-skip-header-csv-reader/
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #next(csvfile)
        for row in spamreader:
            # According to the csv file
            data.t.append((float(row[0])))
            data.x.append((float(row[1])))
            data.y.append((float(row[2])))
            data.z.append((float(row[3])))
    return data


##################################
# Calculate Power Spectral Density
##################################
# def calcPSD(xAxis, yAxis):
#     pass
def calcPSD(xFFT:np.array, yFFT:np.array, fs:float) -> np.array:
    nfft = xFFT.size
    qEven = nfft % 2
    n = (nfft - 2 * qEven) * 2
    psd = (xFFT.conjugate() * yFFT) / (fs * n)
    if qEven:
        psd[1:] *= 2 # Real FFT -> double for non-zero freq
    else: # last point unpaired in Nyquist freq
        psd[1:-1] *= 2 # Real FFT -> double for non-zero freq
    return psd


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
def windowfft(data, M, sample_freq, type):
    samples = len(data)

    num_of_windows = len(data) // M 
    
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
        next_window = data[i:i+M]
        windows.append(next_window)

    spectrums = np.zeros(N+1)

    denominator = hann_window.sum() * sample_freq
    for window in windows:        
        A = np.fft.rfft(window*hann_window)
        spectrum = calcPSD(A, A, sample_freq).real
        spectrums += spectrum

    final_thing = spectrums/len(windows) * num_of_windows
    
    return final_thing


##################################
# Calculate significant wave height
##################################
def getSWH():
    pass



#takes two arrays representing the x (frequency) and y (PSD) coordinates for an acceleration spectrum density graph. 
#returns the PSD array cleaned up
def clean_up(data):
    newArr = list(data)

    #getting the AS(12deltaf) and AS(24deltaf)
    ASF12 = np.argmin(np.abs(np.array([i[0] for i in newArr])-0.01))
    ASF24 = np.argmin(np.abs(np.array([i[0] for i in newArr])-0.02))

    #print out AS(12deltaf) and AS(24deltaf) information for testing
    #print("ASF12: \n\tFrequency = ", (newArr[ASF12][0]), "\n\tPSD = ", (newArr[ASF12][1]), "\n", "ASF24: \n\tFrequency =", (newArr[ASF24][0]), "\n\tPSD = ", (newArr[ASF24][1]))

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



