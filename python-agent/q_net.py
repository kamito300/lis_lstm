# -*- coding: utf-8 -*-

import copy
import numpy as np
from chainer import Chain, cuda, FunctionSet, Variable, optimizers
import chainer.functions as F
import chainer.links as L
from predict_action_model import PredictActionModel
from predict_scene_model import PredictSceneModel


class QNet:

    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**3  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5  # Data size of history. original: 10^6
    hist_size = 10 #original: 4

    def __init__(self, use_gpu, enable_controller, dim):
        self.use_gpu = use_gpu
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller
        self.dim = dim

        print("Initializing Q-Network...")

        hidden_dim = 256

        self.action_model = PredictActionModel(self.dim, hidden_dim, self.num_of_actions)
        self.scene_model = PredictSceneModel(self.dim)

        if self.use_gpu >= 0:
            self.model.to_gpu()

        self.model_target = copy.deepcopy(self.model)

        self.action_optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.action_optimizer.setup(self.action_model)

        self.scene_optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.scene_optimizer.setup(self.scene_model)

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    def action_forward(self, state, action, reward, state_dash, episode_end):
        num_of_batch = state.shape[0]

        for i in xrange(len(state[0])):
            s = Variable(state[:, i, :])
            q = self.q_func(s)  # Get Q-value

        # Generate Target Signals
        for i in xrange(len(state_dash[0])):
            s_dash = Variable(state_dash[:, i, :])
            tmp = self.q_func_target(s)

        if self.use_gpu >= 0:
            tmp = list(map(np.max, tmp.data.get()))  # max_a Q(s',a)
        else:
            tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)

        max_q_dash = np.asanyarray(tmp, dtype=np.float32)

        if self.use_gpu >= 0:
            target = np.asanyarray(q.data.get(), dtype=np.float32)
        else:
            # make new array
            target = np.array(q.data, dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = reward[i] + self.gamma * max_q_dash[i]
            else:
                tmp_ = reward[i]

            action_index = self.action_to_index(action[i])
            target[i, action_index] = tmp_

        # TD-error clipping
        if self.use_gpu >= 0:
            target = cuda.to_gpu(target)
        td = Variable(target) - q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = np.zeros((self.replay_size, self.num_of_actions), dtype=np.float32)
        if self.use_gpu >= 0:
            zero_val = cuda.to_gpu(zero_val)
        zero_val = Variable(zero_val)
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, q

    def scene_forward(self, state, state_dash):
        for i in xrange(len(state[0])):
            s = Variable(state[:, i, :])
            next_scene = self.scene_model(s)  # Get Scene-value
        loss = F.mean_squared_error(next_scene - state_dash)
        return loss
        


    def stock_experience(self, time,
                        state, action, reward, state_dash,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
        else:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
            self.d[3][data_index] = state_dash
        self.d[4][data_index] = episode_end_flag

    def experience_replay(self, time):
        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.d[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.d[1][replay_index[i]]
                r_replay[i] = self.d[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.d[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.d[4][replay_index[i]]

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_replay)

            # Scene Model update
            self.scene_model.reset_state()
            self.scene_optimizer.zero_grads()
            scene_loss = self.scene_forward(s_replay, s_dash_replay)
            scene_loss.backward()
            self.scene_optimizer.update()
        
            # Gradient-based update
            self.action_model.reset()
            self.action_optimizer.zero_grads()
            loss, _ = self.action_forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.action_optimizer.update()

    def q_func(self, state):
        q = self.action_model(state)
        #h4 = F.relu(self.model.l4(state))
        #q = self.model.q_value(h4 / 255.0)
        return q

    def q_func_target(self, state):
        q = self.action_model_target(state)
        #h4 = F.relu(self.model_target.l4(state / 255.0))
        #q = self.model_target.q_value(h4)
        return q

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        q = self.q_func(s)
        q = q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print(" Random"),
        else:
            if self.use_gpu >= 0:
                index_action = np.argmax(q.get())
            else:
                index_action = np.argmax(q)
            print("#Greedy"),
        return self.index_to_action(index_action), q

    def e_greedy_with_interest(self, state, epsilon, last_state):
        index_action, q = e_greedy(state, epsilon)
        # return 0 to 1
        interest = sigmoid(self.scene_model(last_state, state))
        #if sigmoid(self.scene_model(last_state, state)) > 0.5:
        #    interest = true
        #else:
        #    interest = false
        return index_action, q, interest

        
    def sigmoid(z):
        return 1 / (1 + math.e**(-z)) 
        

    def target_model_update(self):
        self.action_model_target = copy.deepcopy(self.action_model)

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)
