# -*- coding: utf-8 -*-

import copy
import numpy as np
from chainer import Chain, cuda, FunctionSet, Variable, optimizers
import chainer.functions as F
import chainer.links as L

# RNN model
class PredictSceneModel(Chain):
    def __init__(self, in_dim, out_dim):
        super(PredictSceneModel, self).__init__(
            l = L.LSTM(in_dim, out_dim),
            l2 = L.Lenear(out_dim, out_dim),
        )

    def to_cpu(self):

    def to_gpu(self, device):
    
    def reset_state(self):j

    def __call__(self, x):
        h = self.l(x)
        out = self.l2(x)
        return out 

    def interest(self, x):
        return F.sum(abs(self(x) - x))
        

    def reset(self):
      self.l.reset_state()
