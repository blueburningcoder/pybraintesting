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
netFile = "net3.p"
batchSize = 10
learningRate = 0.035
net = buildNetwork(784, HiddenNeurons, 10)
debug = True

import os.path

if os.path.isfile(netFile) and not debug:
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

trainer = BackpropTrainer(net, ds, learningRate, verbose = True)
random.seed()

# training the network for one epoch FIXME: for some reason no noticeable effect
def Training():
    # trainer = BackpropTrainer(net, ds) # useful at all?
    print "training with training rate %0.3f" % learningRate
    print trainer.train()

# testing as to how correct the net currently is -> usually about ~10% at initial
def evaluate():
    print "testing"
    correct = 0
    correctVal = 0
    for test, ans in testData:
        if ans == np.argmax(net.activate(list(test) ) ):
            correct += 1
    for test, ans in validationData:
        if ans == np.argmax(net.activate(list(test) ) ):
            correctVal += 1
    return correct, correctVal


for i in range(numEpochs): 
    # printing the current status as to how good it is right now
    (res1, res2) = evaluate()
    percent = (res1 + res2) / (len(testData) + (len(validationData) / 1.0) )
    percent *= 100
    print ("Epoch %02d: got %d/%d and %d/%d digits correct, %0.2f percent" % (i, res1, len(testData), res2, len(validationData), percent) )
    # writing progress to file
    with open(netFile, "wb") as f:
        pickle.dump(net, f)
    # training the net further
    Training()

endres1, endres2 = evaluate()
print ("Finish: got %d/%d and %d/%d digits correct." % (endres1, len(testData), endres2, len(validationData) ) )
