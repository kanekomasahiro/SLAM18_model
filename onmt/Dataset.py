from __future__ import division

import math
import copy
import torch
from torch.autograd import Variable

import onmt
from collections import defaultdict


class Dataset(object):

    def __init__(self, datas, cuda, volatile=False):
        def wrap(b):
            if b is None:
                return b
            if cuda:
                b = b.cuda()
            b = Variable(b, volatile=volatile)
            return b

        for data in datas:
            for key, value in data.items():
                data[key] = [wrap(v) for v in value]

        self.datas = datas

        #self.cuda = cuda

        #self.volatile = volatile
        self.numBatches = len(datas)

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        return self.datas[index]

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
