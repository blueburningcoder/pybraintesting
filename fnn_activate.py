# coding: utf-8
for i in range(20):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(), trndata['class'])
    tstresult = percentError(trainer.testOnClassData(dataset=tstdata),tstdata['class'])
    print "epoch: %4d" % trainer.totalepochs,  \
        "  train error: %5.2f%%" % trnresult,  \
        "  test error: %5.2f%%" % tstresult
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1) # the highest output activation gives the class
    out = out.reshape(X.shape)
    figure(1)
    ioff() # interactive graphics off
    clf() # clear the plot
    hold(True) # overplot on
    for c in [0,1,2]:
        here, _ = where(tstdata['class'] == c)
        plot(tstdata['input'][here, 0], tstdata['input'][here,1], 'o')
    if out.max() != out.min():  # safety check against flat field
        contourf(X,Y,out)  # plot the contour
    ion()  # interactive graphics on
    draw() # update the plot 
    ioff()
    show()
