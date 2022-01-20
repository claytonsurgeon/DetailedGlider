import math
import numpy as np


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


#takes two arrays representing the x (frequency) and y (PSD) coordinates for an acceleration spectrum density graph. 
#returns the Displacement spectrum density data from AS
def getDS(OF, T):
    F = list(OF[0:len(OF)])

    #loop through data and apply the displacement spectrum density conversion to every PSD value. 
    for i in range(len(F)): 
        if(OF[i] > 0):
            FAS = OF[i]
            F[i] = FAS/(math.pow((2*math.pi*T[i]), 2))
            
    return F


#calculates the area under the graph which should be significant wave height
def getSH(data):
    SWH = 0 #significant wave height variable

    #running through the data, calculating small rectangles, and summing them up
    for i in range(len(data)-1): 
        SWH += (data[i+1][0] - data[i][0]) * (data[i][1]) 
    return SWH
