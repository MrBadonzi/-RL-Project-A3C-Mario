import math
from os import mkdir
from os.path import join
from random import random, randrange
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from all_in_one_env import wrap_environment,from_tuple_to_tensor
from model import CNNDQN
import torch.nn.functional as F
from collections import deque
from copy import copy
from gym.wrappers import Monitor


def process_train(id, global_model, global_optimizer, args,eps_start=1.0, eps_final=0.01, epsilon_decay = 100000):
    torch.manual_seed(123 + id) #each process is initialized with a seed for reproduction + its id so that they explore different things
    loss_function = nn.MSELoss()
    writer = SummaryWriter(args.log_path)
    env=wrap_environment(args.environment,args.action_space)
    model = CNNDQN(env.observation_space.shape[0], len(args.action_space))
    if args.transfer:
        model_name = join(args.saved_path, '%s.dat' % args.environment)
        model.load_state_dict(torch.load(model_name))
    state = torch.from_numpy(env.reset())
    if args.gpu:
        torch.cuda.manual_seed(123 + id)
        model.cuda()
        state = state.cuda()
    model.train()
    done = True
    curr_episode = 0
    for episode in range(args.num_episodes):
        curr_episode += 1
        model.load_state_dict(global_model.state_dict()) #Synchronize thread-specific parameters to global ones
        states = []
        next_states=[]
        rewards = []
        actions = []
        dones=[]
        while True:
            epsilon = eps_final + (eps_start - eps_final) * math.exp(-1 * ((episode + 1) / epsilon_decay))
            if random() > epsilon:
                if args.gpu:
                    state = state.cuda()
                q_value = model(state)
                action = q_value.max(1)[1].item()
            else:
                action = randrange(env.action_space.n)

            next_state, reward, done, info = env.step(action)
            state = torch.from_numpy(next_state)
            if args.gpu:
                state = state.cuda()
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            if done:
                state = torch.from_numpy(env.reset())
                if args.gpu:
                    state = state.cuda()
                break

        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1)
        states = from_tuple_to_tensor(states)
        next_states = from_tuple_to_tensor(next_states)
        dones = torch.IntTensor(dones).reshape(-1, 1)

        qvals = model(states)
        qvals = torch.gather(qvals, 1, actions)

        next_qvals = model(next_states)
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        target_qvals = rewards + (1 - dones) * args.gamma * next_qvals_max
        total_loss = loss_function(qvals, target_qvals)
        writer.add_scalar("Train_{}/Loss".format(id), total_loss, curr_episode)
        writer.add_scalar("Train_{}/Reward".format(id), reward, curr_episode)
        global_optimizer.zero_grad()
        total_loss.backward()
        for local_param, global_param in zip(model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad  #TODO Asynchronous update of θ using dθ and of θv using dθv

        global_optimizer.step()




def process_test(global_model, args):
    torch.manual_seed(123 + args.num_processes)
    best_reward = -float('inf')
    best_average = -float('inf')
    rewards = []
    index = 0
    saved_models = []
    env = wrap_environment(args.environment, args.action_space)
    model = CNNDQN(env.observation_space.shape[0], len(args.action_space))
    if args.transfer:
        model_name = join(args.saved_path, '%s.dat' % args.environment)
        model.load_state_dict(torch.load(model_name))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    episode_reward = 0.0
    curr_step = 0
    curr_episode=0
    actions = deque(maxlen=args.max_actions)
    action_list=[]
    while True:
        curr_step += 1
        flag=False
        if done:
            model.load_state_dict(global_model.state_dict())
        if args.gpu:
            state = state.cuda()
        qvals = model(state)
        action = qvals.max(1)[1].item()
        next_state, reward, done, info = env.step(action)
        if args.render:
            env.render()
        actions.append(action)
        episode_reward += reward
        action_list.append(action)
        if info['flag_get']:
            print('Reached the flag!')
            flag = True
        if done or actions.count(actions[0]) == actions.maxlen:
            done = True
            index+=1
            rewards.append(episode_reward)
            save_model = False
            save_flag = False
            if curr_episode % args.save_interval == 0 and curr_episode > 0:
                torch.save(model.state_dict(),join("checkpoints", '%s.dat' % args.environment))
            curr_episode+=1
            if episode_reward > best_reward:
                best_reward = episode_reward
                print('New high score of %s! Saving model' % round(episode_reward, 3))
                save_model = True
            average=sum(rewards[-100:]) / len(rewards[-100:]) #this index is to keep into account the most recent rewards
            if average > best_average:
               best_average = average
               print('New best average reward of %s! Model is improving' % round(best_average, 3))
            if flag:
                model_already_saved=False
                for modl in saved_models:
                    if modl == action_list:
                        model_already_saved = True
                if not model_already_saved:
                    copied_list = copy(action_list)
                    saved_models.append(copied_list)
                save_flag = not model_already_saved
                save_model = True
            if save_model:
                record_best_run(model, args, index)
                torch.save(model.state_dict(),join(args.saved_path, '%s.dat' % args.environment))
            if save_flag:
                try:
                    mkdir('saved_models/run%s' % index)
                except:
                    pass
                torch.save(model.state_dict(),join('saved_models/run%s' % index,'%s.dat' % args.environment))
            print('Episode %s - Reward: %s, Best: %s, Average: %s' % (index, round(episode_reward, 3),round(best_reward, 3),round(average, 3)))

            episode_reward = 0.0
            actions.clear()
            action_list.clear()
            next_state = env.reset()
        state = torch.from_numpy(next_state)




def record_best_run(model, args, episode): #record video
    def record(episode_id):
        return True

    env = wrap_environment(args.environment, args.action_space)
    env = Monitor(env, 'recordings/run%s' % episode, force=True, video_callable=record)
    # Update the framerate to 20 frames per second for a more naturally-paced playback.
    env.metadata['video.frames_per_second'] = 20.0

    #Only exploitation for the recording
    episode_reward = 0.0
    state = torch.from_numpy(env.reset())
    while True:
        qvals = model(state)
        action = qvals.max(1)[1].item()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = torch.from_numpy(next_state)
        if done:
            return