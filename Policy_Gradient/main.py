"""
This is the main file for Policy gradient implementation in continuous and discrete space
"""
import gym
import ipdb
from  policy_gradient import  PolicyGradient

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0") # CartPole-v0")
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        action_space_type = "discrete"
        print("Discrete action space")
    elif type(env.action_space) == gym.spaces.box.Box:
        action_space_type = "continuous"
        print("Continuous action space")


    model = PolicyGradient(env, action_space_type, max_ep_len= env._max_episode_steps)
    model.train()
