# coding: utf-8
from pybrain.tools.shortcuts import buildNetwork
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

netFile = "net.p"

global net

# if os.path.isfile(netFile):
with open(netFile, "rb") as f:
    net = pickle.load(f)    
print "net loaded"
# else:
#    net = buildNetwork(784, HiddenNeurons, 10)
#    print "net was not existent so it got created"

# net got either loaded or constructed, either way it's 'save' to use it now


# creating the plot to draw on
image = np.zeros( (28, 28) )

fig = plt.figure()
ax  = fig.add_subplot(111)
im  = ax.imshow(image, vmin = 0, vmax = 1)

def lininp(old, new):
    if old == None:
        yield new
        return
    for kappa in np.arange(0, 1, 0.01):
        yield (1-kappa)*old + kappa*new

def stripzero(str):
    if str[0] == '0':
        return str[1:]
    else:
        return str


oldpos = None
run = 1
def onmouse(event):
    global image, ax, im, oldpos, run
    if event.button == 1:
        # for every point on the connected line set matrice-entry to one
        for p in lininp(oldpos, np.array([event.xdata, event.ydata])):
            image[int(round(p[1])),   int(round(p[0]))]   = 1
            image[int(round(p[1]-1)), int(round(p[0]))]   = 1
            image[int(round(p[1]+1)), int(round(p[0]))]   = 1
            image[int(round(p[1])),   int(round(p[0]-1))] = 1
            image[int(round(p[1])),   int(round(p[0]+1))] = 1
            oldpos = np.array([event.xdata, event.ydata])
        # actualize Graphical output
        im.set_data(image)
        ax.draw_artist(im)
        im.figure.canvas.blit(im.figure.bbox)

        # send picture through neuronal net
        # act = feedforward(image.reshape(784,1))[-1]
        act = net.activate(image.reshape(784) )
        print "sending through net ... try %d " % run
        run += 1 
        print("%d" % np.argmax(act) + ": " + ", ".join([ stripzero("%1.4f" % a) for a in act ]))
    else:
        oldpos = None
        # Reset picture at right click
        if event.button == 3:
            image = np.zeros( (28, 28) )
            im.set_data(image)
            ax.draw_artist(im)
            im.figure.canvas.blit(im.figure.bbox)


fig.canvas.mpl_connect('button_press_event', onmouse)
fig.canvas.mpl_connect('motion_notify_event', lambda event: onmouse(event) )


plt.show()
