#!/usr/bin/python

import numpy as np
from chainer import Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import math

word2id = {}
input = []
for line in open('develop-en.dat'):
    tmp = []
    words = line.rstrip().split()
    for word in words:
        if word2id.has_key(word):
            tmp.append(word2id[word])
        else:
            word2id[word] = len(word2id)
            tmp.append(word2id[word])
    input.append(tmp)

print 'Sentence = %d, Vocabulary = %d' % (len(input), len(word2id))
vocabulary = len(word2id)
codesize = 400
hiddensize = 200

model = FunctionSet(
    embed = F. EmbedID(vocabulary, codesize),
    x_to_h = F.Linear(codesize, hiddensize),
    h_to_y = F.Linear(hiddensize, vocabulary),
)
optimizer = optimizers.Adam()
optimizer.setup(model)

def forward(x_data, y_data):
    x = model.embed(Variable(x_data))
    t = Variable(y_data)
    h = F.tanh(model.x_to_h(x))
    y = F.tanh(model.h_to_y(h))
    return F.softmax_cross_entropy(y, t)

for epoch in range(100):
    indexes = np.random.permutation(len(input))
    log_perplexity = np.zeros(())
    for index in indexes:
        if len(input[index]) < 2: continue
        loss = forward(np.array(input[index][:-1], dtype=np.int32), np.array(input[index][1:], dtype=np.int32))
        log_perplexity += loss.data.reshape(())
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()
    print 'Epoch: %d | Perplexity = %f' % (epoch+1, math.exp(log_perplexity/len(indexes)))
