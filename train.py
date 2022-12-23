from os import mkdir
from os.path import join
import torch
from tensorboardX import SummaryWriter
from all_in_one_env import wrap_environment
from model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from copy import copy
from gym.wrappers import Monitor


def process_train(id, global_model, global_optimizer, args):
    torch.manual_seed(123 + id) #each process is initialized with a seed for reproduction + its id so that they explore different things
    writer = SummaryWriter(args.log_path)
    env=wrap_environment(args.environment,args.action_space)
    model = ActorCritic(env.observation_space.shape[0], len(args.action_space))
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
        if done:
            hx = torch.zeros((1, 512), dtype=torch.float)
            cx = torch.zeros((1, 512), dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
        if args.gpu:
            hx = hx.cuda()
            cx = cx.cuda()
        log_policies = []
        values = []
        rewards = []
        entropies = []
        while True:
            logits, value, hx, cx = model(state, hx, cx) #logits is result estimating Q value from Actor, value=state-value from the Critic
            policy = F.softmax(logits, dim=1) #policy probabilities from actor
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True) #entropy forces the actor to consider as much actions as possible while still maximizing the reward
            multinomial = Categorical(policy) #assigns probabilities to each action (policy)
            action = multinomial.sample().item() #action with greatest probability is most likely chosen
            next_state, reward, done, info = env.step(action)
            state = torch.from_numpy(next_state)
            if args.gpu:
                state = state.cuda()
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)
            if done:
                state = torch.from_numpy(env.reset())
                if args.gpu:
                    state = state.cuda()
                break
        R = torch.zeros((1, 1), dtype=torch.float) #reaches here only if done thereafter we are at St so R=0
        gae = torch.zeros((1, 1), dtype=torch.float)
        if args.gpu:
            R = R.cuda()
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * args.gamma * args.tau
            gae = gae + reward + args.gamma * next_value.detach() - value.detach() #Gae= gae + R − V (si; θ′v)
            next_value = value                                                     #where R = ri + γ*R
            R = R * args.gamma + reward      #update of R --------------------------------^
            actor_loss = actor_loss + log_policy * gae  #TODO Accumulate gradients wrt θ′
            critic_loss = critic_loss + (R - value) ** 2 / 2  #TODO Accumulate gradients wrt θv′
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - args.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(id), total_loss, curr_episode)
        global_optimizer.zero_grad()
        total_loss.backward()
        for local_param, global_param in zip(model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        global_optimizer.step()




def process_test(global_model, args):
    torch.manual_seed(123 + args.num_processes)
    best_reward = -float('inf')
    best_average = -float('inf')
    rewards = []
    index = 0
    saved_models = []
    env = wrap_environment(args.environment, args.action_space)
    model = ActorCritic(env.observation_space.shape[0], len(args.action_space))
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
        with torch.no_grad():
            if done:
                hx = torch.zeros((1, 512), dtype=torch.float)
                cx = torch.zeros((1, 512), dtype=torch.float)
            else:
                hx = hx.detach()
                cx = cx.detach()
        logit, value, hx, cx = model(state, hx, cx)
        policy = F.softmax(logit, dim=1)
        action = torch.argmax(policy).item() #action with greatest probability is chosen, thereafter exploitation only
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
                for model in saved_models:
                    if model == action_list:
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
                mkdir('saved_models/run%s' % index)
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
    hx = torch.zeros((1, 512), dtype=torch.float)
    cx = torch.zeros((1, 512), dtype=torch.float)
    #TODO include steps for second so it does not get to the end of the timer if stuck
    while True:
        logit, value, hx, cx = model(state, hx, cx)
        policy = F.softmax(logit, dim=1)
        action = torch.argmax(policy).item()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = torch.from_numpy(next_state)
        hx = hx.detach()
        cx = cx.detach()
        if done:
            return