import cv2
import numpy as np
from gym import make, ObservationWrapper, Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace



class FrameDownsample(ObservationWrapper):
    def __init__(self, env):
        super(FrameDownsample, self).__init__(env)
        self._width = 84
        self._height = 84
        self.observation_space = Box(low=0,high=255, #image pixels
                                     shape=(self._width, self._height, 1), #image dimension
                                     dtype=np.uint8)

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) #from colored to gray image
        frame = (cv2.resize(frame,(self._width, self._height), #resize to fixed size
                            interpolation=cv2.INTER_AREA))[:, :, None] #The choice is INTER_AREA because we are shrinking the image to a 84x84
        return frame




class ImageToPyTorch(ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0,high=1.0,shape=(obs_shape[::-1]),dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class FrameBuffer(ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrameBuffer, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self._num_steps = num_steps
        self.observation_space = Box(obs_space.low.repeat(num_steps, axis=0),obs_space.high.repeat(num_steps, axis=0),dtype=self._dtype)

    def step(self, action):
        states = []
        state, total_reward, done, info = self.env.step(action)
        for _ in range(self._num_steps):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(self._dtype), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self._num_steps)], 0)
        return states[None, :, :, :].astype(self._dtype)


class NormalizeFloats(ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0



class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']: #wins the stage
                reward += 350.0
            else:
                reward -= 50.0 #episode ended because the character died or time is over
        return state, reward / 10.0, done, info


#TODO Fix this mess : too many classes
def wrap_environment(environment, action_space):
    env = make(environment)
    env = JoypadSpace(env, action_space)
    env = FrameDownsample(env)
    env = ImageToPyTorch(env)
    env = FrameBuffer(env, 4)
    env = NormalizeFloats(env) #last step of preprocessing : normalizes observations
    env = CustomReward(env)
    return env