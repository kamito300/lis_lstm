# -*- coding: utf-8 -*-

import copy
import numpy as np
from chainer import Chain, cuda, FunctionSet, Variable, optimizers
import chainer.functions as F
import chainer.links as L

# RNN model
class PredictActionModel(Chain):
    def __init__(self, in_dim, out_dim, num_of_actions):
        super(PredictActionModel, self).__init__(
            l = L.LSTM(in_dim, out_dim),
            q_value = L.Linear(out_dim, num_of_actions),
        )

    def __call__(self, x):
        h = self.l(x)
        q = self.q_value(h)
        return q

    def reset(self):
      self.l.reset_state()
