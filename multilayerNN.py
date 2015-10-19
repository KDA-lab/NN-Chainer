#!/usr/bin/python

import numpy as np
from chainer import Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F

nsize = 2
model = FunctionSet(
    x_to_h = F.Linear(2, nsize),
    h_to_y = F.Linear(nsize, 1),
)

def forward(x_data, y_data):
    x = Variable(x_data)
    t = Variable(y_data)
    h = F.sigmoid(model.x_to_h(x))
    y = F.sigmoid(model.h_to_y(h))
    return F.mean_squared_error(y, t)

optimizer = optimizers.Adam()
optimizer.setup(model)

datasize = 4
batchsize = 2

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

for epoch in range(20000):
#    print 'Epoch %d' % epoch
    indexes = np.random.permutation(datasize)
    sum = 0.0
    for i in range(0, datasize, batchsize):
        x_batch = np.asarray(x[indexes[i:i+batchsize]])
        y_batch = np.asarray(y[indexes[i:i+batchsize]])
        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        sum += loss.data
        loss.backward()
        optimizer.update()
    if ((epoch + 1) % 1000) == 0: print epoch+1, ":", sum/datasize
