import math
from os import mkdir
from os.path import join
from random import random, randrange
import torch
from tensorboardX import SummaryWriter
from torch import nn
from all_in_one_env import wrap_environment
from SarsaModel import CNNDQN
from collections import deque
from copy import copy
from gym.wrappers import Monitor
from random import seed


def epsilon_greedy(state, epsilon, args, num_actions, hx, cx, model):
    if random() > epsilon:
        if args.gpu:
            state = state.cuda()
        q_value, hx, cx = model(state, hx, cx)
        action = q_value.max(1)[1].item()
    else:
        action = randrange(num_actions)

    return action

writer = SummaryWriter("tensorboard/a3c_super_mario_bros")

def process_train(id, global_model, global_optimizer, args, eps_start=1.0, epsilon_decay=100000,
                  Itarget=5, IasincUpdate=100):
    torch.manual_seed(
        123 + id)  # each process is initialized with a seed for reproduction + its id so that they explore different things
    loss_function = nn.MSELoss()

    env = wrap_environment(args.environment, args.action_space)
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

    seed(123 + id)
    gen = random()

    if gen < 0.4:
        eps_final = 0.1
    elif gen < 0.7:
        eps_final = 0.01
    else:
        eps_final = 0.5

    for episode in range(args.num_episodes):

        curr_step = 0
        curr_episode += 1
        global_optimizer.zero_grad()
        model.load_state_dict(global_model.state_dict())  # Synchronize thread-specific parameters to global ones

        hx = torch.zeros((1, 512), dtype=torch.float)
        cx = torch.zeros((1, 512), dtype=torch.float)

        if args.gpu:
            hx = hx.cuda()
            cx = cx.cuda()



        while True:

            epsilon = eps_final + (eps_start - eps_final) * math.exp(-1 * ((episode + 1) / epsilon_decay))

            # epsilon greedy
            action = epsilon_greedy(state=state, epsilon=epsilon, args=args, num_actions=env.action_space.n, hx=hx,
                                    cx=cx, model=model)

            qvals, hx, cx = model(state, hx, cx)
            qvals = qvals[0][action]

            next_state, reward, done, info = env.step(action)
            state = torch.from_numpy(next_state)

            if args.gpu:
                state = state.cuda()

            next_action = epsilon_greedy(state=state, epsilon=epsilon, args=args, num_actions=env.action_space.n, hx=hx,
                                         cx=cx, model=model)

            next_qvals, hx, cx = model(state, hx, cx)
            next_qvals = next_qvals[0][next_action]
            target_qvals = reward + (1 - done) * args.gamma * next_qvals.detach()
            total_loss = loss_function(qvals, target_qvals)



            total_loss.backward(retain_graph=True)

            curr_step += 1

            if curr_step % Itarget == 0:
                global_optimizer.step()

            if curr_step % IasincUpdate == 0:
                for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                    if global_param.grad is not None:
                        break
                    global_param._grad = local_param.grad  # TODO Asynchronous update of θ using dθ and of θv using dθv
                global_optimizer.zero_grad()

            if done:
                state = torch.from_numpy(env.reset())
                if args.gpu:
                    state = state.cuda()
                for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                    if global_param.grad is not None:
                        break
                    global_param._grad = local_param.grad  # TODO Asynchronous update of θ using dθ and of θv using dθv

                hx = hx.detach()
                cx = cx.detach()

                break


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
    curr_episode = 0
    hx = None
    cx = None
    actions = deque(maxlen=args.max_actions)
    action_list = []
    while True:
        curr_step += 1
        flag = False
        if done:
            model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                hx = torch.zeros((1, 512), dtype=torch.float)
                cx = torch.zeros((1, 512), dtype=torch.float)
            else:
                hx = hx.detach()
                cx = cx.detach()
        if args.gpu:
            state = state.cuda()
        qvals, hx, cx = model(state, hx, cx)
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
            index += 1
            rewards.append(episode_reward)
            save_model = False
            save_flag = False
            if curr_episode % args.save_interval == 0 and curr_episode > 0:
                torch.save(model.state_dict(), join("checkpoints", '%s.dat' % args.environment))
            curr_episode += 1
            if episode_reward > best_reward:
                best_reward = episode_reward
                print('New high score of %s! Saving model' % round(episode_reward, 3))
                save_model = True
            average = sum(rewards[-100:]) / len(
                rewards[-100:])  # this index is to keep into account the most recent rewards
            if average > best_average:
                best_average = average
                print('New best average reward of %s! Model is improving' % round(best_average, 3))
            if flag:
                model_already_saved = False
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
                torch.save(model.state_dict(), join(args.saved_path, '%s.dat' % args.environment))
            if save_flag:
                try:
                    mkdir('saved_models/run%s' % index)
                except:
                    pass
                torch.save(model.state_dict(), join('saved_models/run%s' % index, '%s.dat' % args.environment))
            print('Episode %s - Reward: %s, Best: %s, Average: %s' % (
                index, round(episode_reward, 3), round(best_reward, 3), round(average, 3)))


            writer.add_scalar("Train_{}/Reward".format(id), episode_reward, index)

            episode_reward = 0.0
            actions.clear()
            action_list.clear()
            next_state = env.reset()
        state = torch.from_numpy(next_state)


def record_best_run(model, args, episode):  # record video
    def record(episode_id):
        return True

    env = wrap_environment(args.environment, args.action_space)
    env = Monitor(env, 'recordings/run%s' % episode, force=True, video_callable=record)
    # Update the framerate to 20 frames per second for a more naturally-paced playback.
    env.metadata['video.frames_per_second'] = 20.0

    # Only exploitation for the recording
    episode_reward = 0.0
    state = torch.from_numpy(env.reset())
    hx = torch.zeros((1, 512), dtype=torch.float)
    cx = torch.zeros((1, 512), dtype=torch.float)
    while True:
        qvals, hx, cx = model(state, hx, cx)
        action = qvals.max(1)[1].item()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = torch.from_numpy(next_state)
        hx = hx.detach()
        cx = cx.detach()
        if done:
            return
