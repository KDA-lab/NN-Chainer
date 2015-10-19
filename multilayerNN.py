#!/usr/bin/python

import numpy as np
from chainer import Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
datasize = len(x)
hsize = 2

model = FunctionSet(
    x_to_h = F.Linear(2, hsize),
    h_to_y = F.Linear(hsize, 1),
)
optimizer = optimizers.Adam()
optimizer.setup(model)

def forward(x_data, y_data):
    x = Variable(x_data)
    t = Variable(y_data)
    h = F.sigmoid(model.x_to_h(x))
    y = F.sigmoid(model.h_to_y(h))
    return y, F.mean_squared_error(y, t)

batchsize = 2
for epoch in range(20000):
    indexes = np.random.permutation(datasize)
    sum = np.zeros(())
    for i in range(0, datasize, batchsize):
        x_batch = np.asarray(x[indexes[i:i+batchsize]])
        y_batch = np.asarray(y[indexes[i:i+batchsize]])
        optimizer.zero_grads()
        output, loss = forward(x_batch, y_batch)
        sum += loss.data.reshape(())
        loss.backward()
        optimizer.update()
    if ((epoch+1) % 1000) == 0: print 'Epoch %d: Error = %f' % (epoch+1, sum/datasize)

output, loss = forward(x, y)
print output.data
