import argparse
import os
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT,RIGHT_ONLY,SIMPLE_MOVEMENT)
import torch
from all_in_one_env import wrap_environment
from model import ActorCritic
import torch.nn.functional as F
from gym.wrappers import Monitor

Action_space_choices = {
    'right_only': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT
}

os.environ['OMP_NUM_THREADS'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--environment', type=str, help = f"The OpenAI gym environment to use. Default: SuperMarioBros-1-1-v0",default = "SuperMarioBros-1-1-v0")
    parser.add_argument('--action_space', choices=Action_space_choices,
                        help="Specify the action space to use as given by gym-super-mario-bros. Default : complex",
                        default=Action_space_choices["complex"])
    parser.add_argument("--saved_path", type=str, default="best_models/A3c_Model/SuperMarioBros-1-1-v0.dat")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument('--gpu', action='store_true', help = "Specify this parameter to run on GPU. Default: False")
    args = parser.parse_args()
    return args

def test(args):
    torch.manual_seed(123)
    env = wrap_environment(args.environment, args.action_space)
    env = Monitor(env, f"{args.output_path}/{args.environment}", force=True, video_callable=lambda episode_id: True)# Update the framerate to 20 frames per second for a more naturally-paced playback.
    env.metadata['video.frames_per_second'] = 20.0
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(args.saved_path, map_location={'0':'dml'}))
    if args.gpu:
        model.cuda()
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True

    while True:
        if done:
            hx = torch.zeros((1, 512), dtype=torch.float)
            cx = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            hx = hx.detach()
            cx = cx.detach()
        if args.gpu:
            hx = hx.cuda()
            cx = cx.cuda()
            state = state.cuda()

        logits, value, hx, cx = model(state, hx, cx)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item() #Only exploitation
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["flag_get"]:
            print("Level completed!")



if __name__ == "__main__":

    args = get_args()
    test(args)
