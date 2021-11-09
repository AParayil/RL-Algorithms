"""
This is the main program for Reinforce implementation in continuous and discrete action space
This implementation is a modified version of https://github.com/VXU1230/Medium-Tutorials/tree/master/policy_gradient
Options: Continuous or discrete action space
"""
import gym
import ipdb
from policy_gradient import  Reinforce

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        action_space_type = "discrete"
        print("Discrete action space")
    elif type(env.action_space) == gym.spaces.box.Box:
        action_space_type = "continuous"
        print("Continuous action space")

    model = Reinforce(env, action_space_type, max_ep_len= env._max_episode_steps)
    model.train()

