from random import random, randrange
import numpy as np
import torch
from torch import nn


class CNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.reshape((-1, 64*7*7))
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon, args, num_actions):
        if random() > epsilon:
            if args.gpu:
                state = state.cuda()
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = randrange(num_actions)
        return action