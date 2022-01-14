
import matplotlib.pyplot as plt # importing module for creating a graph

import csv
from helpers import * # importing helper module

import sys





def main(argv):
    i = 0
    #initial AS
    data = [] #2d array for all the x and y values. Format = [[frequency1, PSD1], [frequency2, PSD2], [frequency3, PSD3], ...., [frequency1, PSD1]]
    
    #reading csv file
    with open(argv[0], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if(len(row)<6):
                continue
            
            if(row[0] == "Period" or float(row[0]) <= 0):
                continue
    
            if(len(row)>6 and i < 1026):
                data.append([1/float(row[0]), float(row[7])])
                i+=1

            else:
                break
    
    #sorting all the data by frequency values to make the graph look ok
    sorted_date=sorted(data, key=lambda k: [k[0], k[1]])
    

    #plotting now sorted data
    plot1 = plt.figure(1)
    plt.plot([i[0] for i in sorted_date], [i[1] for i in sorted_date], color='purple', linewidth=.5)   
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD ((ms^(-2))^(2)/Hz)')
    plt.title('Acceleration Spectrum Density (U)')

    #cleaned up AS with data dependent noise function
    data2 = clean_up(sorted_date)

    #plotting cleaned up AS 
    plot2 = plt.figure(2)
    plt.plot([i[0] for i in data2], [i[1] for i in data2],color='blue', linewidth=.5)     
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD ((ms^(-2))^(2)/Hz)')
    plt.title('Cleaned Up Acceleration Spectrum Density (U)')

    #Getting the Displacement spectrum
    data3 = getDS(data2)
    
    #plotting the displacement spectrum
    plot3 = plt.figure(3)
    plt.plot([i[0] for i in data3], [i[1] for i in data3], color='green', linewidth=.5)    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD ((m^2)/Hz)')
    plt.title('Displacement Spectrum Density (U)')

    
    #was outputting data to test integration with Ben
    with open('displacement.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data3)

    #getting significant wave height
    datawithRMS = getSH(data3)

    #plotting the significant wave height
    plot3 = plt.figure(4)
    plt.plot([i[0] for i in datawithRMS], [i[2] for i in datawithRMS], color='orange', linewidth=.5)    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RMS')
    plt.title('Significant Wave Height')
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])