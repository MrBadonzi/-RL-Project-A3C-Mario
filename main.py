from argparse import ArgumentParser
from os import environ
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT)
from optimizer import GlobalRMSprop
from Train_A3C import A3C_train
from Train_A3C import A3C_test
from Train_Sarsa import *
from Train_QLearning import *

from A3CModel import *

environ[
    'OMP_NUM_THREADS'] = '1'  # this option avoids python multiprocessing to call another process that is doing multiprocessing

Action_space_choices = {
    'right_only': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT
}

Agent_choices = ['A3C', 'Sarsa', 'Q-learning']

Beta = 0.01
Environment = "SuperMarioBros-1-1-v0"
Gamma = 0.8
Learning_rate = 1e-4
Max_actions = 200
Num_episodes = 12000
Num_processes = 4
Tau = 1.0
Pretrained_models_directory = 'best_models'


def train(args):
    torch.manual_seed(123)
    mp = torch.multiprocessing.get_context('spawn')
    env = wrap_environment(args.environment, args.action_space)

    if args.agent == 'A3C':
        global_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        Pretrained_models_directory = 'best_models/A3c_Model'
        args.saved_path = Pretrained_models_directory

    elif args.agent == 'Sarsa':
        global_model = CNNDQN(env.observation_space.shape[0], env.action_space.n)
        Pretrained_models_directory = 'best_models/Sarsa_Model'
        args.saved_path = Pretrained_models_directory


    else:
        global_model = CNNDQN(env.observation_space.shape[0], env.action_space.n)
        Pretrained_models_directory = 'best_models/Q_Model'
        args.saved_path = Pretrained_models_directory

    if args.transfer:
        model_name = join(Pretrained_models_directory, '%s.dat' % Environment)
        global_model.load_state_dict(torch.load(model_name))
    if args.gpu:
        torch.cuda.manual_seed(123)
        global_model.cuda()
    global_model.share_memory()
    # global_optimizer = GlobalAdam(global_model.parameters(),lr=args.learning_rate)
    global_optimizer = GlobalRMSprop(global_model.parameters(), lr=args.learning_rate)
    processes = []

    if args.agent == 'A3C':
        for index in range(args.num_processes):
            process = mp.Process(target=A3C_train,
                                 args=(index, global_model, global_optimizer, args))  # start training on each process
            process.start()
            processes.append(process)
        process = mp.Process(target=A3C_test, args=(global_model, args))

    elif args.agent == 'Sarsa':
        for index in range(args.num_processes):
            process = mp.Process(target=Sarsa_train,
                                 args=(index, global_model, global_optimizer, args))  # start training on each process
            process.start()
            processes.append(process)
        process = mp.Process(target=Sarsa_test, args=(global_model, args))

    elif args.agent == 'Q-learning':
        for index in range(args.num_processes):
            process = mp.Process(target=Qtrain,
                                 args=(index, global_model, global_optimizer, args))  # start training on each process
            process.start()
            processes.append(process)
        process = mp.Process(target=Qtest, args=(global_model, args))

    process.start()
    processes.append(process)
    for process in processes:
        process.join()  # wait for the thread to complete then manually stop the process


if __name__ == "__main__":
    parser = ArgumentParser(description='')
    # TODO reinsert action in argument?
    parser.add_argument('--action_space', choices=Action_space_choices,
                        help="Specify the action space to use as given by gym-super-mario-bros. Default : complex",
                        default=Action_space_choices["complex"])
    parser.add_argument('--beta', type=float, help=f"The coefficient used in the entropy calculation. Default : {Beta}",
                        default=Beta)
    parser.add_argument('--environment', type=str, help=f"The OpenAI gym environment to use. Default: {Environment}",
                        default=Environment)
    parser.add_argument('--gpu', action='store_true', help="Specify this parameter to run on GPU. Default: False")
    parser.add_argument('--gamma', type=float,
                        help=f"Specify the discount factor to use for rewards. Default : {Gamma}", default=Gamma)
    parser.add_argument('--learning_rate', type=float, help=f"The learning rate to use. Default: {Learning_rate}",
                        default=Learning_rate)
    parser.add_argument('--max_actions', type=int,
                        help=f"Specify the maximum number of actions to repeat while in the testing phase. Default: {Max_actions}",
                        default=Max_actions)
    parser.add_argument('--num_episodes', type=int,
                        help=f"The number of episodes to run in the given environment. Default: {Num_episodes}",
                        default=Num_episodes)
    parser.add_argument('--num_processes', type=int,
                        help=f"The number of training processes to run in parallel. Default: {Num_processes}",
                        default=Num_processes)
    parser.add_argument('--render', action='store_true',
                        help="Specify if rendering. Note that rendering scenes will lower the learning speed. Default: False")
    parser.add_argument('--tau', type=float, help=f"The value used to calculate GAE. Default : {Tau}", default=Tau)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--saved_path", type=str, default=Pretrained_models_directory)
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")

    parser.add_argument("--agent", type=str, help="specify what algorithm you want to run", default=Agent_choices[2])

    parser.add_argument('--transfer', action='store_true',
                        help="Transfer model weights from a previously-trained model. Default: False")

    args = parser.parse_args()
    train(args)
