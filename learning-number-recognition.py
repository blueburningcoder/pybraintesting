# coding: utf-8
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import cPickle as pickle
import numpy as np
import mnist_loader
import random

# useful for something betweeen 30 and 100
HiddenNeurons = 40
# number of training-iterations over the database, and tests inbetween
numEpochs = 30
# file the net should be saved in
netFile = "net0.p"
learningRate = 0.043        # best for SigmoidLayer and 40 Neurons is 0.043 -> ~92%
learningRateDecay = 1.0 #0.95 # the usual one is 1.0 so why does it go wrong with sth. slightly below?
Momentum = 0.025  # the amount as to which the gradient descent from the last timestamp is used
weightDecay = 0.015  # the amount as to which the weights are decreased
net = buildNetwork(784, HiddenNeurons, 10, bias = True, hiddenclass = SigmoidLayer)
debug = True

import os.path


def getFileName():
    name = netFile
    it = 0
    while (True):
        if os.path.isfile(name):
            name = "net%d.p" % it
            it += 1
        else:
            return name


if os.path.isfile(netFile) and not debug:
    print "loading already existing ANN"
    with open(netFile, "rb") as f:
#        global net
        net = pickle.load(f)
else:
#    global net
    print "creating new Network, %d neurons hidden" % HiddenNeurons


# creating the supervised dataset
ds = SupervisedDataSet(784, 10)

trainingData, validationData, testData = mnist_loader.load_data_wrapper()

# initiating / loading the dataset in the right format for pyBrain
def initTestDataSet():
    print "initiating dataset"
    for pic, ans in trainingData:
        ds.addSample(list(pic), list(ans) )

initTestDataSet()

trainer = BackpropTrainer(net, ds, learningrate = learningRate, lrdecay=learningRateDecay, momentum = Momentum, verbose = True, weightdecay = weightDecay)
# random.seed()

# training the network for one epoch 
def Training():
    print "training with learning rate %0.3f and decay %0.3f, a momentum of %0.3f and a weightdecay of %0.3f" % (learningRate, learningRateDecay, Momentum, weightDecay)
    trainer.train()

# testing as to how correct the net currently is for two different sets -> usually about ~10% at initial
def evaluate():
    print "---- ---- --- Testing --- ---- ---- (%s)" % netFile
    correct = 0
    correctVal = 0
    for test, ans in testData:
        if ans == np.argmax(net.activate(list(test) ) ):
            correct += 1
    for test, ans in validationData:
        if ans == np.argmax(net.activate(list(test) ) ):
            correctVal += 1
    return correct, correctVal


netFile = getFileName()

for i in range(numEpochs): 
    # printing the current status, including test results and training status
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
percent = (endres1 + endres2) / (len(testData) + (len(validationData) / 1.0) )
percentstr = str(percent * 100) + '%'
print ("Finish: got %d/%d and %d/%d digits correct. (%s) (%s)" % (endres1, len(testData), endres2, len(validationData), percentstr, netFile) )
