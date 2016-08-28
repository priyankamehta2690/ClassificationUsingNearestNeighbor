import csv
import urllib2
import math
import random
import numpy as np
import timeit
import sklearn

def main():
    testY=[]
    correct=0
    incorrect=0
    condensedIdx=[]
    runningTime=0.0

    trainX,trainY,testX,originaltestY=getData()
    algo=input("Enter 1 for basic kNN algorithm or Enter 2 for condensed kNN algorithm: ")
    k=input("Enter the value of k: ")
    N=input("Enter the number of training subsamples: ")
    for i in range(N):
        randomIndex = random.randint(0,len(trainX)-1)
        trainX[i]=trainX[randomIndex]
        trainY[i]=trainY[randomIndex]
    if(algo==1):
        start = timeit.default_timer()
        testY = testknn(trainX, trainY, testX, k) 
        stop = timeit.default_timer()
        runningTime=stop - start
    if(algo==2):
        start = timeit.default_timer()
        condensedIdx= condensedata(trainX, trainY)
        testY = testknn(condensedIdx[0], condensedIdx[1], testX, k)  
        stop = timeit.default_timer()
        runningTime=stop - start
    for i in range(len(testY)):
        if(testY[i][0] == originaltestY[i]):
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    print runningTime
    print correct
    #print sklearn.metrics.confusion_matrix(originaltestY, testY)
    
def testknn(trainX, trainY, testX, k): 
    neighbors=[]
    testY=[]
    for i in range(len(testX)):
        neighbors.append(getNeighbor(trainX,trainY,testX[i],k))
        testY.append(calculateLabel(neighbors[i]))
    return testY

def condensedata(trainX,trainY):
    subsetX=[]
    subsetY=[]
    minSubsetX=[]
    minSubsetY=[]
    outputY=[]
    for i in range(len(trainX)):
        randomIndex = random.randint(0,len(trainX)-1)
        subsetX.append(trainX[randomIndex])
        subsetY.append(trainY[randomIndex])
    minSubsetX.append(subsetX[0])
    minSubsetY.append(subsetY[0])
    subsetX.remove(subsetX[0])
    subsetY.remove(subsetY[0])
    outputY=testknn(minSubsetX,minSubsetY,subsetX,1)
    i=0
    while(i<len(subsetX)):
        sampleX = []
        sampleX.append(subsetX[i])
        outputY = testknn(minSubsetX, minSubsetY, sampleX, 1)
        if(outputY[0] != subsetY[i]):
            minSubsetX.append(subsetX[i])
            minSubsetY.append(subsetY[i])
        i=i+1
        continue
    return (minSubsetX,minSubsetY)
    
def calculateLabel(neighbor):
	classVotes = {}
	for x in range(len(neighbor)):
		response = neighbor[x]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=getKey, reverse=True)
	return sortedVotes[0][0]  
    
def getData():
    dataSet=[]
    labelData=[]
    trainX=[]
    trainY=[]
    testX=[]
    originaltestY=[]
    data = csv.reader(urllib2.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"))
    for row in data:
        dataSet.append(row)
    for item in dataSet:
        labelData.append(item[0])
        del(item[0])
    for item in range(15000):
        trainX.append(dataSet[item])
        trainY.append(labelData[item])
    for item in range(15000,20000):
        testX.append(dataSet[item])
        originaltestY.append(labelData[item])
    for x in range(len(trainX)):
        for y in range(16):
            trainX[x][y]=float(trainX[x][y])
    for x in range(len(testX)):
        for y in range(16):
            testX[x][y]=float(testX[x][y])
    trainX=np.array(trainX)
    testX=np.array(testX)
    return trainX,trainY,testX,originaltestY   
    
def euclideanDistance(testItem,trainItem):
    d=0
    for k in range(16):
        d += math.pow((testItem[k]-trainItem[k]),2)
    return math.sqrt(d)
    
def getNeighbor(trainX,trainY,testItem,k):
    eucDistance=[]
    neighbor=[]
    for j in range(len(trainX)):
        eucDistance.append((trainY[j], euclideanDistance(testItem,trainX[j])))    
    sortedEucDistance=sorted(eucDistance,key=getKey)
    for i in range(k):
        neighbor.append(sortedEucDistance[i][0])
    return neighbor
     
def getKey(item):
    return item[1]    
     
if __name__=="__main__":
    main()