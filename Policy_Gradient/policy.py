"""
Corresponds to the policy implemetation for both discrete and continuous action space
Glorot_uniform initialization of the network layers seems to give better results in continuous action space
"""
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import  Dense, Input
from tensorflow.keras.models import Model
import  sklearn.preprocessing
import  numpy as np


class Policy():
    def __init__(self, input_size, output_size, action_space_type, env):
        self.action_space_type = action_space_type
        self.env = env

        if self.action_space_type == "continuous":
            """
            This is to scale the states 
            """
            self._scaler = sklearn.preprocessing.StandardScaler()
            observation_examples = np.array([env.observation_space.sample() for x in range(1000)])
            self._scaler = sklearn.preprocessing.StandardScaler()
            self._scaler.fit(observation_examples)

        if self.action_space_type == "discrete":
            self.model = keras.Sequential(
                layers=[
                    keras.Input(shape=(input_size,)),
                    layers.Dense(64, activation="relu", name="relu_layer"),
                    layers.Dense(output_size, activation="linear", name="linear_layer")
                ],
                name="policy")

        elif self.action_space_type == "continuous":
            inputs = Input(shape=(input_size,), name='state')
            layer1 = Dense(40, activation="elu", kernel_initializer= 'glorot_uniform',
                           name="elu_layer1")(inputs)
            layer2 = Dense(40, activation="elu",kernel_initializer= 'glorot_uniform',
                           name="elu_layer2")(layer1)
            mean = Dense(output_size,
                         kernel_initializer='glorot_uniform',
                         name='mean')(layer2)

            stddev = Dense(output_size,
                           kernel_initializer= 'glorot_uniform',
                           name='stddev')(layer2)
            stddev = K.softplus(stddev) + 1e-5




            # use of softplusk avoids stddev = 0

            self.model = Model(inputs, [mean, stddev], name='policy')








    def action_distribution(self, observations):
        if self.action_space_type =="discrete":
            logits = self.model(observations)
            return tfp.distributions.Categorical(logits=logits)
        elif self.action_space_type == "continuous":


            [loc, scale] = self.model(self._scaler.transform(observations))
            """
            For actions in multi-dimension, use tfp.distributions.MultivariateNormalDiag()
            """

            return tfp.distributions.Normal(loc, scale)

    def sampl_action(self, observations):
        sampled_actions = self.action_distribution(self._scaler.transform(observations)).sample()
        if self.action_space_type == "continuous":
            action = K.clip(sampled_actions,
                        self.env.action_space.low[0],
                        self.env.action_space.high[0]).numpy()
            return action
        return sampled_actions.numpy()






