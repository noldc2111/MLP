import scipy as sp
import re, os, string, csv
import random as r

"""Casey Nold;  nold@pdx.edu; CS545 HW 2, MultiLayer Perceptron"""

def main():
    
    #Retrieving test and training files
    trainingPath = "/Users/caseynold/Desktop/CS545/ML HW 1/trainingData/"
    testPath = "/Users/caseynold/Desktop/CS545/ML HW 1/testData/"
    trainingFiles = dirList(trainingPath)
    testFiles = dirList(testPath)

    # Allow for user to choose parameters
    hiddenNodes = int(input("Enter the number of Hidden Nodes:"))
    momentum = float(input("Enter the Momentum:"))
    learningRate = float(input("Enter the learning Rate:"))
    outputNodes = 26

    # Generate 26 lists of target values ranked from .9 to .1
    target = targetGeneration()

    # Generate random weight for the input and hidden layers
    randomInputWeights  = randomWeightVector(hiddenNodes,17) 
    randomHiddenWeights = randomWeightVector(outputNodes,hiddenNodes+1)

    # initialize the epoch counter and training accuracy  
    epoch = 0
    trainingAcc = 0
    
    while(epoch <= 100):
        # list to write values to file
        dataVar = []
        # initialize correct/incorrect counters for train/test sets
        trainCor,trainInc,testCor,testInc = 0,0,0,0

        print("EPOCH: ", epoch)

        # loop over each trainingfile
        for each in range(len(trainingFiles)):
            #read the data in from file
            trainingData = readFile(trainingFiles[each],trainingPath)
            testingData = readFile(testFiles[each],testPath)
            
            #take the training data and place into a matrix
            matrixTrain = turnMatrix(trainingData)
            matrixTest = turnMatrix(testingData)
            
            # standardized data using x = (x-µ)/σ where x is a value in the data set
            # µ is the mean and σ is the standard deviation
            standardizedTraining, standardizedTest = dataStandardization(matrixTrain, matrixTest)
            
            # ascertain the dimensions of the test/training data and shuffle the matirces
            i,j = standardizedTraining.shape
            sp.random.shuffle(standardizedTraining[1])
            l,m = standardizedTest.shape
            sp.random.shuffle(standardizedTest[1])
            
    
            for x in range(i):
                # train each data class
                outputActivations, hiddenActivations,totalTrainError = feedForward(standardizedTraining[x],randomInputWeights,randomHiddenWeights,hiddenNodes,outputNodes,target[each])
                # temporarily store weight values to use when running test data through the MLP
                testInputWeight = randomInputWeights
                testHiddenWeight = randomHiddenWeights

                # run backpropagation and get new weights
                randomInputWeights, randomHiddenWeights = training(randomInputWeights,randomHiddenWeights,learningRate,momentum,outputActivations, standardizedTraining[x],hiddenActivations, target[each])
                #calculate the number of correct and incorrect classifications for the training example and add these values to the total correct/incorrect counters
                correctTrain,incorrectTrain = accumulator(outputActivations,target[each])
                trainCor += correctTrain
                trainInc += incorrectTrain
            
            for test in range(l):
                # run the testData through the feedForward and obtain the accuracy 
                testOutActivations, testHiddenActivations, totalTestError = feedForward(standardizedTest[test],testInputWeight,testHiddenWeight,hiddenNodes,outputNodes,target[each])
                correctTest, incorrectTest = accumulator(testOutActivations,target[each])
                testCor += correctTest
                testInc += incorrectTest
        # calculate the training accuracy        
        trainingAcc = float("{0:.0f}".format((trainCor/(trainCor+trainInc))*100))
        # add the values of correct/incorrect values and epoch number to a list to be exported to excel spreadsheet
        dataVar.append(epoch)
        dataVar.append(trainCor)
        dataVar.append(trainInc)
        dataVar.append(testCor)
        dataVar.append(testInc)
        
        # open excel sheet and append values from each epoch
        with open("hw2_trial2.csv",'a') as f:
            writer = csv.writer(f, delimiter =',')
            writer.writerow(dataVar)
        # increment the epoch
        epoch = epoch+1


def accumulator(outputActivation,target):
    """ Take a list of output activations and a list of target values
        and accumulate the number of correct versus incorrect classifications.
        Return the correct and incorrect accumulations"""
    #initialize the accumulators
    correct = 0
    incorrect = 0

    # loop through lists and check for equality, adding 1 to either
    # the correct or incorrect accumulator depending on the equality
    #print("Target and outputAct:",target, outputActivation)

    for each in range(0,len(outputActivation)):
        if(target[each] == outputActivation[each]):
            correct = correct + 1
        else: incorrect = incorrect + 1
    total = incorrect + correct
    acc = (correct/total)
    
    return correct, incorrect
    
def targetGeneration():
    """ Generates and reurns  a list of lists containing target values with values
        of either 0.9 or 0.1. The 0.9 shifts positions to work with each
        respective training example."""
    
    # create a list of size 26
    targets = []*26
    # empty list to hold each target list
    actualTargets = []
    # values of the list
    high = 0.9
    low = 0.1
    # add a 0.1 in each position of the list. 
    # then substitute a 0.9 in the position 
    # indicated by the index of the first for
    # loop.
    for i in range(0,26):
        for j in range(0,26):
            targets.append(low)
        targets[i] = high
        actualTargets.append(targets)
        targets = []
    return actualTargets


def training(inputToHiddenWeights,hiddenToOutputWeights,learningRate,momentum,outputActivation,inputVector,hiddenActivation,target):
    """ Backpropagate the activations to calculate the new values of the weights of the input and hidden layer. 
        Return the changed weights for the input and hidden layer"""
    
    # find the dimensions of the weight matrices
    i,j = inputToHiddenWeights.shape
    k,l = hiddenToOutputWeights.shape

    # creating two new matries for the updated weight values
    deltaWeightInputToHidden = sp.zeros(i*j).reshape(i,j)
    deltaWeightHiddenToOutput = sp.zeros(k*l).reshape(k,l)
    
    # Step one: find how much the total error changes with respect to the output activations. aka ∂Etot/∂out = (output - target)
    errorChange = []# length 26
    for each in range(len(outputActivation)):
        errorChange.append(-(target[each] - outputActivation[each]))

    # Step two: find how the output changes with respect to the total network input. aka ∂out/∂net
    outputChange = [] #length 26
    for each in range(len(outputActivation)):
        outputChange.append((outputActivation[each] *(1-outputActivation[each])))

    # Step three: multiply step one and two with the respective hidden layer activation( ∂net/∂w == out_h)
    errorTotalwRespectToWeight = [] #130  --> 5 hidden nodes( 4 + 1 bias) * 26 output nodes
    for x in range(len(hiddenActivation)):
        for y in range(len(outputActivation)):
            errorTotalwRespectToWeight.append((errorChange[y] * outputChange[y] * hiddenActivation[x]))
    # Step 4: calculate the new weights leading to the outputs from the hidden layer ∆w = wᵢ - η *α *∂  ---> wᵢ: weight in question;  η: learning rate; α: momentum 
    for x in range(0,k):
        for y in range(0,l):
            deltaWeightHiddenToOutput[x,y] = (hiddenToOutputWeights[x,y]) - (learningRate * errorTotalwRespectToWeight[l] * momentum)

    # Step 5: calculate the errorChange * outputChange
    outputErrorOverNetInput = [] # length 26
    for each in range(len(errorChange)):
        outputErrorOverNetInput.append(errorChange[each] * outputChange[each])

    # Step 6: Find the sum of total error over the output
    totalErrorOverHiddenOutput = 0
    for a in range(0,i):
        for b in range(0,j):
            totalErrorOverHiddenOutput += outputErrorOverNetInput[b] * inputToHiddenWeights[a,b] # outputErrorOverNetInput[a]
            
    # Step 7: calculate the hidden acivation output change with respect to net input. aka out_h*(1-out_h)
    hiddenChange = [] #list length 5
    for each in range(len(hiddenActivation)):
        hiddenChange.append((hiddenActivation[each] * (1-hiddenActivation[each])))

    # Step 8: multiply step 6 and 7 times the weights from the input to the hidden layer
    errorTotalPerWeight = sp.zeros(i*j).reshape(i,j)
    for a in range(0,i):
        for b in range(0,j):
            errorTotalPerWeight[a,b] = totalErrorOverHiddenOutput * hiddenChange[a] * inputToHiddenWeights[a,b]
            
    # Step 9: calculate the new weights leading from the input layer to the hidden layer            
    for x in range(0,i):
        for y in range(0,j):
            deltaWeightInputToHidden[x,y] = inputToHiddenWeights[x,y] - (learningRate * errorTotalPerWeight[x,y] * momentum)
    
    
    return deltaWeightInputToHidden,deltaWeightHiddenToOutput

def feedForward(inputVector,randomInputWeights,randomHiddenWeights,hiddenNodes,outputNodes,target):
    """ Take a set of inputs, weights, number of hidden units and outputs. Calculate the 
         activations of the hidden and output layer. Return the activations of the hidden 
         and output layers"""

    # initialize the lists for hidden and output activations
    hiddenActivation = []
    outputActivation = []
    totalError  = 0

    # take each input and the weight associated with this input and calculate the 
    # dot product. 
    for x in range(hiddenNodes): 
        hiddenActivation.append(sigmoid((sp.dot(inputVector,randomInputWeights[x]))))
    # append the bias...
    hiddenActivation.append(1) 
    
    # for each output, calculate the activation. 
    for w in range(outputNodes):
            outputActivation.append(float("{0:.2f}".format(sigmoid((sp.dot(hiddenActivation,randomHiddenWeights[w]))))))
    # Etotal = ∑ 1/2*(targetᵢ - outputᵢ)^2
    for each in range(outputNodes):
        error = .5 * (sp.square((target[each] - outputActivation[each])))
        totalError += error 
        
    return outputActivation, hiddenActivation, totalError

def sigmoid(z):
    """ Take an activation and minimize the loss. Return the sigmoid value"""
    sigma = 1/ (1+sp.exp(-z))
    return sigma

def genInitWeights(num):
    """ Generate a set of random weights between -.25 and 25.
        Return an array ofrandom weights"""
    initWeight = sp.zeros(num).reshape(num,)   #initialize list for weights
    #for i in range(0,16):
    for i in range(num):
        initWeight[i] = r.uniform(-.25,.25) #generate random weights, w= [-.25,.25]
    return initWeight

def dataStandardization(trainingMatrix, testMatrix): 
    """take two nxm matrices A and B and do standard normal distribution to
        each value in the matrix A and use these values to normalize matrix B. Returns two normalized nxm matrices"""
    i, j = trainingMatrix.shape
    normalizedTrainMatrix = sp.zeros(i*j).reshape(i,j)
    normalizedTestMatrix = sp.zeros(i*j).reshape(i,j)
    
    for x in range(i):
        for y in range(j):
            meanCol = sp.mean(trainingMatrix[x])
            stdCol = sp.std(trainingMatrix[x])
            dataTrainStand = (trainingMatrix[x,y] - meanCol)/stdCol
            dataTestStand = (testMatrix[x,y] - meanCol)/stdCol
            normalizedTrainMatrix[x,y] = dataTrainStand
            normalizedTestMatrix[x,y] = dataTestStand
            normalizedTrainMatrix[x,16] = 1
            normalizedTestMatrix[x,16] = 1
            
    return normalizedTrainMatrix, normalizedTestMatrix


def turnMatrix(inputVals):
    """takes a list of lists and converts to a matrix of nxm where
        n is the number of lists and m is the number of values in each list.
        Returns an nxm matrix"""
    listLength = len(inputVals)
    listDepth = len(inputVals[0])
    inputMatrix = sp.zeros(listLength*listDepth).reshape(listLength,listDepth)
    for i in range(listLength):
        subList = inputVals[i]
        for j in range(listDepth):
            inputMatrix[i,j] = subList[j]
    return inputMatrix

def readFile(fileName, filePath):
    """ Given a file name and a filepath, stream the numeric values and place into a list. Return the list--
        these are the input values"""
    inputVals = []
    file = filePath + fileName   #concatenate the filepath with the file name
    with open(file, 'r') as file:# stream the file
        fileline = file.readlines()   #read a line from the file
        regex = r"[-+]?\d*\.\d+|[-+]?\d+"   # regular expression to obtain float values
        for each in fileline:
            digit = re.findall(regex, each)   # add the floats to a temorary list
            digit.append(float('1')) #adding the bias
            inputVals.append(list(map(float,digit))) # add to the final list of inputs.
        return inputVals
    
def randomWeightVector(i,j):
    """ Take dimensions of an array and create a matrix of random weights of dimension i,j.
         Return the random weights"""
    biasWeight = genInitWeights(1)
    randomWeights  = sp.zeros(i*j).reshape(i,j)
    for x in range(i):
        weightVector = genInitWeights(j+1)
        for y in range(j):
            randomWeights[x,y] = weightVector[y]
        randomWeights[x,j-1] = biasWeight
    return randomWeights

def dirList(path):
    """ Take a file path and return a list of the files in that directory"""
    
    directory = os.listdir(path) #return the names of the files in the directory
    fileList = [] # initialize a list for the filenames
    #iterate through the directory and append each file to the fileList
    for file in directory:
        if file != ".DS_Store" and file != "perceptron.py": # there are some files we don't want to have in our list...
            fileList.append(file) #add the file to the list
    return fileList #return the file list
     
