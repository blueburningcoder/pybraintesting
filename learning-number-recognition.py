# coding: utf-8
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import cPickle as pickle
import numpy as np
import mnist_loader

# useful for something betweeen 30 and 100
HiddenNeurons = 40
# number of training-iterations over the database
numEpochs = 30
trainLength = 100
# file the net should be saved in
netFile = "net.p"
batchSize = 10
eta = 3.0
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
    net = buildNetwork(784, HiddenNeurons, 10)


# creating the supervised dataset
ds = SupervisedDataSet(784, 10)

trainingData, validationData, testData = mnist_loader.load_data_wrapper()

def initTestDataSet():
    print "initiating dataset"
    for pic, ans in trainingData:
        ds.addSample(list(pic), list(ans) )


initTestDataSet()
trainer = BackpropTrainer(net, ds)

def Training():
    print "training"
    for i in range(trainLength):
        print trainer.train()
        print "%d / %d" % (i+1, trainLength)

def evaluate(testingData):
    print "testing"
    correct = 0
    for test, ans in testingData:
        if ans == np.argmax(net.activate(list(test) ) ):
            correct += 1
    return correct


for i in range(numEpochs): 

    print ("Epoch %02d: %d von %d Ziffern richtig erkannt." % (i, evaluate(testData), len(testData) ) )

    with open(netFile, "wb") as f:
        pickle.dump(net, f)

    Training()
