import os, sys

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyper Parameters
BATCH_SIZE = 32
LR = 5e-7  # learning rate
EPSILON = 1  # greedy policy
GAMMA = 0.95  # reward discount
TARGET_REPLACE_ITER = 50  # target update frequency
MEMORY_CAPACITY = 2000

FEATURES = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, outshape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(FEATURES, 4)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2  = nn.Linear(4,4)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(4, outshape)
        self.fc3.weight.data.normal_(0, 0.1)
        self.to(DEVICE)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


class DQN(object):
    def __init__(self, env):

        self.env = env

        self.N_ACTIONS = env.action_space.n
        self.STATES_SHAPE = env.observation_space.shape
        self.ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

        self.eval_net, self.target_net = Net(outshape=self.N_ACTIONS), Net(outshape=self.N_ACTIONS)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory_capacity = MEMORY_CAPACITY

        self.memory = np.zeros((MEMORY_CAPACITY, FEATURES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = torch.tensor(x, device=DEVICE, dtype=torch.float)

        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            actions_value = actions_value.cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.tensor(b_memory[:, :FEATURES], device=DEVICE, dtype=torch.float)
        b_a = torch.tensor(b_memory[:, FEATURES:FEATURES + 1].astype(int), device=DEVICE, dtype=torch.long)
        b_r = torch.tensor(b_memory[:, FEATURES + 1:FEATURES + 2], device=DEVICE, dtype=torch.float)
        b_s_ = torch.tensor(b_memory[:, -FEATURES:], device=DEVICE, dtype=torch.float)


        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)


        self.optimizer.zero_grad()
        loss.backward()
        # # gradient clipping
        # for param in self.eval_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()