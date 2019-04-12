import retro
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

## # TODO:
    # training
    # increase randomness if top score gets stale
    # familiarity index? bias for exploring something new

class MarioNet():
    def __init__(self):
        self.game_history = [] # [([obs],[reward]), ...]
        self.training_epochs = 2
        self.max_memory_size = 100
        self.discount_factor = 0.8
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Conv2D(filters=4,
                                    kernel_size=80,
                                    input_shape=(224, 240, 3),
                                    data_format="channels_last",
                                    activation="relu"))
        self.net.add(keras.layers.MaxPooling2D())
        self.net.add(keras.layers.Flatten())
        # self.net.add(keras.layers.Dense(, activation='sigmoid'))
        self.net.add(keras.layers.Dense(9))
        self.net.compile(optimizer=tf.train.AdamOptimizer(0.1),
                        loss='mse',
                        metrics=['accuracy'])

    def fire(self, observation):
        out = self.net.predict(np.array([observation]))[0]
        return [1 if x > 0.5 else 0 for x in out]

    def train(self):
        print('--training--')
        inputs = []
        desired_outputs = []
        for game_i, game in enumerate(self.game_history):
            max_steps = len(game[1]) # idk if len of rewards is faster
            predictions = self.net.predict(np.array(game[0]))
            for step_i, observation in enumerate(game[0]):
                max_q = max(predictions[step_i+1]) # output of outlook for next netState
                reward = game[1] + self.discount_factor * max_q
                desired_outputs.append(predictions[step_i] * reward)

        inputs = np.array(inputs)
        desired_outpus = np.array(desired_outpus)
        if desired_outpus.size != 0:
            self.net.fit(inputs, desired_outpus, epochs=self.traning_epochs)

    def on_game_end(self, game):
        self.game_history.append(game)
        if len(self.game_history) > self.max_memory_size:
            self.game_history.pop(0)
        self.train()

mario = MarioNet()

env = retro.make(game='SuperMarioBros-Nes')
obs = env.reset()
obs_list = []
reward_list = []
while True:
    action = mario.fire(obs)
    obs, reward, done, info = env.step(action)
                            # [b ? ? ? ? ? l r a]
    env.render()
    print(reward, action, info)
    obs_list.append(obs)
    reward_list.append(reward)

    if done:
        obs = env.reset()
        mario.on_game_end((obs_list, reward_list))
        obs_list = []
        reward_list = []
env.close()
