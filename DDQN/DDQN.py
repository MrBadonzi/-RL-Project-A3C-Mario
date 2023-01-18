from copy import deepcopy
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from Q_network import Q_network

#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=UserWarning)


class DDQN_agent:

    def __init__(self, env, rew_thre, buffer, learning_rate=0.001, initial_epsilon=0.5, batch_size= 64):

        self.env = env


        self.network = Q_network(env, learning_rate)
        self.target_network = deepcopy(self.network)
        self.buffer = buffer
        self.epsilon = initial_epsilon
        self.batch_size = batch_size
        self.window = 50
        self.reward_threshold = rew_thre
        self.initialize()
        self.step_count = 0
        self.episode = 0


    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        if mode == 'explore':
                action = self.env.action_space.sample()
        else:
                action = self.network.greedy_action(torch.FloatTensor(self.s_0))

        #simulate action
        s_1, r, done, info = self.env.step(action)

        #put experience in the buffer
        self.buffer.append(self.s_0, action, r, done, s_1)

        self.rewards += r

        self.s_0 = s_1.copy()

        self.step_count += 1
        if done:

            self.s_0 = self.env.reset()
        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=15000,
              network_update_frequency=10,
              network_sync_frequency=200):
        self.gamma = gamma
        self.loss_function = nn.MSELoss()
        self.s_0 = self.env.reset()
        self.writer = SummaryWriter("tensorboard")
        self.load_models()

        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
        ep = 8000
        training = True
        self.populate = False
        best_reward = 0
        while training:
            self.s_0 = self.env.reset()

            self.rewards = 0
            done = False
            while not done:
                self.env.render()
                if ((ep % 500) == 0):
                    torch.save(self.network.state_dict(), join("checkpoints", f'{ep}_SuperMarioBros-1-1-v0.dat'))

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                else:
                    done = self.take_step(mode='exploit')
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update(ep)
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    if self.epsilon >= 0.05:
                        self.epsilon = self.epsilon * 0.7
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    mean_loss = np.mean(self.training_loss[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    if self.rewards>best_reward:
                        best_reward=self.rewards
                        torch.save(self.network.state_dict(), 'best_models/SuperMarioBros-1-1-v0.dat')
                    print(
                        "\rEpisode {:d} Best Reward {:.2f} Mean Rewards {:.2f} Episode reward = {:.2f} mean loss = {:.2f}\t\t".format(
                            ep, best_reward,mean_rewards, self.rewards, mean_loss))
                    self.writer.add_scalar("Train_{}/Reward".format(id), self.rewards, ep)
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(ep))
                        #break
        # save models
        self.save_models()

    def save_models(self):
        torch.save(self.network.state_dict(), 'best_models/SuperMarioBros-1-1-v0.dat')

    def load_models(self,eval=False):
        self.network.load_state_dict(torch.load('best_models/SuperMarioBros-1-1-v0.dat'))
        #self.network = torch.load('best_models/SuperMarioBros-1-1-v0.dat')
        if eval:
            self.network.eval()

    def calculate_loss(self, batch,episode):
        #extract info from batch
        states, actions, rewards, dones, next_states = list(batch)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1)
        dones = torch.IntTensor(dones).reshape(-1, 1)
        states = from_tuple_to_tensor(states)
        next_states = from_tuple_to_tensor(next_states)

        ###############
        # DDQN Update #
        ###############
        # Q(s,a) = ??
        qvals = self.network.get_qvals(states)
        qvals = torch.gather(qvals, 1, actions)

        next_qvals= self.target_network.get_qvals(next_states)
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        target_qvals = rewards + (1 - dones)*self.gamma*next_qvals_max

        # loss = self.loss_function( Q(s,a) , target_Q(s,a))
        loss = self.loss_function(qvals, target_qvals)
        self.writer.add_scalar("Train_{}/Loss".format(id), loss, episode)

        return loss


    def update(self,episode):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch,episode)

        loss.backward()
        self.network.optimizer.step()

        self.update_loss.append(loss.item())

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0

    def evaluate(self, eval_env):
        done = False
        s= eval_env.reset()
        rew = 0
        while not done:
            eval_env.render()
            action = self.network.greedy_action(torch.FloatTensor(s))
            s, r, done, info =eval_env.step(action)
            rew += r

        print("Evaluation cumulative reward: ", rew)


def from_tuple_to_tensor(tuple_of_np):
    tensor = torch.zeros((len(tuple_of_np), tuple_of_np[0].shape[1],tuple_of_np[0].shape[2],tuple_of_np[0].shape[3]))
    for i, x in enumerate(tuple_of_np):
        tensor[i] = torch.FloatTensor(x)
    return tensor