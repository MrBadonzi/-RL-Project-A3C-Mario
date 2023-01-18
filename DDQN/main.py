import argparse
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from DDQN import DDQN_agent
from Experience_replay_buffer import Experience_replay_buffer
from all_in_one_env import wrap_environment

Environment = "SuperMarioBros-1-1-v0"

Action_space_choices = {
    'right_only': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT
}



def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    env = wrap_environment(Environment, Action_space_choices["complex"])
    rew_threshold = 345
    buffer = Experience_replay_buffer()
    agent = DDQN_agent(env, rew_threshold, buffer)

    if args.train:
        agent.train()

    if args.evaluate:
        agent.load_models(eval=True)
        agent.evaluate(env)


if __name__ == '__main__':
    main()