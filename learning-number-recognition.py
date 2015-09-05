# coding: utf-8
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import cPickle as pickle
import numpy as np
import mnist_loader
import random

# useful for something betweeen 30 and 100
HiddenNeurons = 40
# number of training-iterations over the database
numEpochs = 30
trainLength = 100
# file the net should be saved in
netFile = "net.p"
batchSize = 10
net = buildNetwork(784, HiddenNeurons, 10)

import os.path

if os.path.isfile(netFile):
    print "loading already existing ANN"
    with open(netFile, "rb") as f:
#        global net
        net = pickle.load(f)
else:
#    global net
    print "creating new Network"


# creating the supervised dataset
ds = SupervisedDataSet(784, 10)

trainingData, validationData, testData = mnist_loader.load_data_wrapper()

# initiating / loading the dataset in the right format for pyBrain
def initTestDataSet():
    print "initiating dataset"
    for pic, ans in trainingData:
        ds.addSample(list(pic), list(ans) )

initTestDataSet()

trainer = BackpropTrainer(net, ds, learningrate = 0.3, verbose = True)
random.seed()

# training the network for one epoch FIXME: for some reason no noticeable effect
def Training():
    # trainer = BackpropTrainer(net, ds) # useful at all?
    print "training"
    print trainer.train()

# testing as to how correct the net currently is -> usually about ~10% at initial
def evaluate(testingData):
    print "testing"
    correct = 0
    for test, ans in testingData:
        if ans == np.argmax(net.activate(list(test) ) ):
            correct += 1
    return correct


for i in range(numEpochs): 
    # printing the current status as to how good it is right now
    print ("Epoch %02d: got %d from %d digits correct." % (i, evaluate(testData), len(testData) ) )
    # writing progress to file
    with open(netFile, "wb") as f:
        pickle.dump(net, f)
    # training the net further
    Training()

print ("Finish: got %d from %d digits correct." % (evaluate(validationData), len(validationData) ) )
