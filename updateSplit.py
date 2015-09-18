from numpy import random
import numpy as np
from random import sample
from numpy.random import permutation
from pybrain.datasets import ClassificationDataSet


def splitWithProportion2(self, proportion = 0.5):
    """Produce two new datasets, the first one containing the fraction given by `proportion` of the samples."""
    indicies = random.permutation(len(self))
    separator = int(len(self) * proportion)

    leftIndicies = indicies[:separator]
    rightIndicies = indicies[separator:]

    leftDs = ClassificationDataSet(2, 1, nb_classes = 3)
    for i in range(separator):
        leftDs.addSample(list(self['input'][i]), 
                list(self['target'][i]))
        leftDs.append('class', self['class'][i])

    rightDs = ClassificationDataSet(2, 1, nb_classes = 3)
    for i in range(separator, len(self)):
        rightDs.addSample(self['input'][i], self['target'][i])
        rightDs.append('class', self['class'][i])

    return leftDs, rightDs

