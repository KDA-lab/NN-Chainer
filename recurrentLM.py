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

datasize = len(input)
vocabulary = len(word2id)
print 'Sentence = %d, Vocabulary = %d' % (datasize, vocabulary)

codesize = 400
hiddensize = 200

model = FunctionSet(
    embed = F.EmbedID(vocabulary, codesize),
    x_to_h = F.Linear(codesize, hiddensize),
    h_to_h = F.Linear(hiddensize, hiddensize),
    h_to_y = F.Linear(hiddensize, vocabulary),
)
optimizer = optimizers.Adam()
optimizer.setup(model)

def forward_one_step(h, cur_word, next_word):
    i = Variable(np.array([cur_word], dtype=np.int32))
    t = Variable(np.array([next_word], dtype=np.int32))
    x = model.embed(i)
    h = F.tanh(model.x_to_h(x)+model.h_to_h(h))
    y = F.tanh(model.h_to_y(h))
    return h, F.softmax_cross_entropy(y, t)

def forward(sentence):
    loss = Variable(np.zeros((), dtype=np.float32))
    log_perplexity = np.zeros(())

    h = Variable(np.zeros((1, hiddensize), dtype=np.float32))
    for cur_word, next_word in zip(sentence[:-1], sentence[1:]):
        h, new_loss = forward_one_step(h, cur_word, next_word)
        loss += new_loss
        log_perplexity += new_loss.data.reshape(())
    return loss, log_perplexity / (len(sentence) - 1)

for epoch in range(1000):
    log_perplexity = np.zeros(())
    indexes = np.random.permutation(len(input))
    data = 0
    for index in indexes:
        if len(input[index]) < 2: continue
        data += 1
        optimizer.zero_grads()
        loss, new_log_perplexity = forward(input[index])
        log_perplexity += new_log_perplexity
        loss.backward()
        optimizer.update()
    if (epoch % 10) == 0:
        print 'Epoch: %d | Perplexity =%f' % (epoch+1, math.exp(log_perplexity / data))
