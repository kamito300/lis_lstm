# -*- coding: utf-8 -*-

import copy
import numpy as np
from chainer import Chain, cuda, FunctionSet, Variable, optimizers
import chainer.functions as F
import chainer.links as L

# RNN model
class PredictSceneModel(Chain):
    def __init__(self, dim):
        super(PredictSceneModel, self).__init__(
            l = L.LSTM(dim, dim),
            l2 = L.Linear(dim, dim),
        )

    def to_cpu(self):
        super(PredictSceneModel, self).to_cpu()

    def to_gpu(self, device=None):
        super(PredictSceneModel, self).to_gpu(device)
    
    def reset(self):
      self.l.reset_state()

    def __call__(self, x):
        h = self.l(x)
        out = self.l2(h)
        return out 

    def interest(self, x, y):
        return F.sum(abs(self(x) - y))

