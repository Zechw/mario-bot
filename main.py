import retro
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

## # TODO:
    # increase randomness if top score gets stale
    # familiarity index? bias for exploring something new

class MarioNet():
    def __init__(self):
        self.game_history = [] # [([obs], [action], [reward]), ...]
        self.training_epochs = 1
        self.q_epochs = 2
        self.max_memory_size = 100
        self.discount_factor = 0.2
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Conv2D(filters=1,
                                    kernel_size=80,
                                    input_shape=(224, 240, 3),
                                    data_format="channels_last",
                                    kernel_initializer='TruncatedNormal'))
        self.net.add(keras.layers.MaxPooling2D())
        self.net.add(keras.layers.Flatten())
        # self.net.add(keras.layers.Dense(18, activation='sigmoid', kernel_initializer='TruncatedNormal'))
        self.net.add(keras.layers.Dense(9, kernel_initializer='TruncatedNormal'))
        self.net.compile(optimizer=tf.train.AdamOptimizer(0.5),
                        loss='mse',
                        metrics=['accuracy'])

    def fire(self, observation):
        out = self.net.predict(np.array([observation]))[0]
        print(out)
        out = [x + np.random.normal() for x in out]
        print(out)
        return [1 if x > 0 else 0 for x in out]

    def train(self):
        for q_i in range(self.q_epochs):
            print('--training-- q:', q_i+1, '/', self.q_epochs)

            inputs = []
            desired_outputs = []
            for game_i, game in enumerate(self.game_history):
                observations = game[0]
                actions = game[1]
                rewards = game[2]
                predictions = self.net.predict(np.array(observations))
                for step_i, observation in enumerate(observations):
                    try:
                        max_q = max(abs(predictions[step_i+1])) # output of outlook for next state
                    except:
                        max_q = 0
                    reward = rewards[step_i] + self.discount_factor * max_q
                    inputs.append(observation)
                    desired_outputs.append([(x-0.5) * reward for x in actions[step_i]])

            inputs = np.array(inputs)
            desired_outputs = np.array(desired_outputs)
            if desired_outputs.size != 0:
                self.net.fit(inputs, desired_outputs, epochs=self.training_epochs)

    def on_game_end(self, game):
        self.game_history.append(game)
        if len(self.game_history) > self.max_memory_size:
            self.game_history.pop(0)
        self.train()

mario = MarioNet()

env = retro.make(game='SuperMarioBros-Nes')
obs = env.reset()
obs_list = []
action_list = []
reward_list = []
f = 0
while True:
    #normalize obs
    obs = np.multiply(obs, 1/255)

    action = mario.fire(obs)
    obs_list.append(obs)
    action_list.append(action)
    obs, reward, done, info = env.step(action)
                            # [b ? ? ? ? ? l r a]
    env.render()
    print(f, reward, action, info)

    reward_list.append(reward)

    f += 1

    if done:
        obs = env.reset()
        mario.on_game_end((obs_list, action_list, reward_list))
        obs_list = []
        reward_list = []
        f = 0
env.close()
