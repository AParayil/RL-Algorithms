"""
Reinforce implementation in TensorFlow 2. Main function is main.py
"""
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from policy  import  Policy

def export_plot(ys, ylabel, title, filename):
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


class Reinforce(object):
    def __init__(self, env, action_space_type, max_ep_len=200,  num_iterations=300, batch_size=2,  output_path="../results/"):

        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.env = env
        self.action_space_type = action_space_type
        self.observation_dim = self.env.observation_space.shape[0]
        if self.action_space_type == "discrete": self.action_dim = self.env.action_space.n
        elif self.action_space_type == "continuous": self.action_dim = self.env.action_space.shape[0]
        self.gamma = 0.99
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.policy = Policy(input_size=self.observation_dim, output_size=self.action_dim, action_space_type=self.action_space_type, env=self.env)


    def play_games(self, env=None, num_episodes = None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        if not env:
            env = self.env


        while (num_episodes or t < self.batch_size):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.max_ep_len):
                states.append(state)
                action = self.policy.sampl_action(np.atleast_2d(state))[0]
                state, reward, done, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1

                if (done or step == self.max_ep_len-1):

                    episode_rewards.append(episode_reward)
                    break



            path = {"observation": np.array(states),
                    "reward": np.array(rewards),
                    "action": np.array(actions)}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= self.batch_size-1: #num_episodes:

                break

        return paths, episode_rewards

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = []
            reversed_rewards = np.flip(rewards,0)
            g_t = 0
            for r in reversed_rewards:
                g_t = r + self.gamma*g_t
                returns.insert(0, g_t)
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-7)
        return returns




    def update_policy(self, observations, actions, returns):
        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)
        with tf.GradientTape() as tape:
            log_prob = self.policy.action_distribution(observations).log_prob(actions)+ 1e-5
            loss = -tf.math.reduce_mean(log_prob * tf.cast(returns, tf.float32))

        grads = tape.gradient(loss, self.policy.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.policy.model.trainable_weights))



    def train(self):
        all_total_rewards = []
        averaged_total_rewards = []
        averaged_total_rewards_cum = []
        for t in range(self.num_iterations):
            paths, total_rewards = self.play_games()
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            returns = self.get_returns(paths)
            self.update_policy(observations, actions, returns)  # it can be  estimated advantage
            avg_reward = np.mean(total_rewards)
            averaged_total_rewards.append(avg_reward)
            averaged_total_rewards_cum.append(np.mean(averaged_total_rewards[-100:]))
            print("Average reward for batch {}: {:04.2f}".format(t,averaged_total_rewards_cum[-1]))
        print("Training complete")
        np.save(self.output_path+ "rewards.npy", averaged_total_rewards)
        export_plot(averaged_total_rewards, "Reward", "Continuous_Car", self.output_path + "rewards4_0.png")
        np.save(self.output_path + "cum_rewards.npy", averaged_total_rewards_cum)
        export_plot(averaged_total_rewards, "Reward", "Continuous_Car", self.output_path + "rewards4_1.png")

    def eval(self, env, num_episodes=1):
        paths, rewards = self.play_games(env, num_episodes)
        avg_reward = np.mean(rewards)
        print("Average eval reward: {:04.2f}".format(avg_reward))
        return avg_reward

