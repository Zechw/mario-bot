import retro
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pickle
import tempfile
import pdb

## # TODO:
    # increase randomness if top score gets stale
    # familiarity index? bias for exploring something new

class MarioNet():
    def __init__(self):
        self.game_history = [] # [([obs], [action], [reward]), ...]
        self.training_epochs = 1
        self.q_epochs = 2
        self.max_memory_size = 100
        self.discount_factor = 0.95
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Conv2D(filters=1,
                                    kernel_size=80,
                                    input_shape=(224, 240, 3),
                                    data_format="channels_last",
                                    kernel_initializer='zeros'))
        self.net.add(keras.layers.MaxPooling2D())
        self.net.add(keras.layers.Flatten())
        # self.net.add(keras.layers.Dense(18, activation='sigmoid', kernel_initializer='zeros'))
        self.net.add(keras.layers.Dense(9, kernel_initializer='zeros'))
        self.net.compile(optimizer=tf.train.AdamOptimizer(0.5),
                        loss='mse',
                        metrics=['accuracy'])

    def fire(self, observation):
        out = self.net.predict(np.array([observation]))[0]
        # print(out)

        # if 0 bias, go right (probably not the best logic... makes sure that initial randomness stumbles oppon a reward)
        if out[7] == 0:
            out[7] = 100
        if out[8] == 0:
            out[8] = 0.2

        out = [x + (np.random.normal() * 0.1) for x in out]
        # print(out)
        return [1 if x > 0 else 0 for x in out]

    def train(self):
        for q_i in range(self.q_epochs):
            print('--training-- q:', q_i+1, '/', self.q_epochs)

            inputs = []
            desired_outputs = []
            for game_i, game in enumerate(self.game_history):
                observations = self.depickle_path(game[0])
                actions = game[1]
                rewards = game[2]
                print('--predicting-- game:', game_i, '--', len(observations))
                start = time.time()
                predictions = self.net.predict(np.array(observations))
                for step_i, observation in enumerate(observations):
                    try:
                        max_q = max(abs(predictions[step_i+1])) # output of outlook for next state
                    except:
                        max_q = 0
                    reward = rewards[step_i] + (self.discount_factor * max_q)
                    inputs.append(observation)
                    desired_outputs.append([(((x-0.5) * reward) - 0.1) for x in actions[step_i]])
                print('  runtime:', time.time()-start)

            print('--fitting--')
            inputs = np.array(inputs)
            desired_outputs = np.array(desired_outputs)
            if desired_outputs.size != 0:
                self.net.fit(inputs, desired_outputs, epochs=self.training_epochs)

    @staticmethod
    def depickle_path(path):
        with open(path, 'rb') as f:
            return pickle.load(f)



    def on_game_end(self, game):
        with tempfile.NamedTemporaryFile(dir='./tmpdir', delete=False) as tmp_file:
            # pdb.set_trace()
            pickle.dump(game[0], tmp_file)
            tmp_file.flush()
            path = tmp_file.name
        self.game_history.append( (path, game[1], game[2]) )
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
g = 0
while True:
    action = mario.fire(obs)

    obs_list.append(obs)
    action_list.append(action)
    obs, reward, done, info = env.step(action)
                            # [b ? ? ? ? ? l r a]
    env.render()
    print(g, f, reward, action, info)

    reward_list.append(reward)

    # bundle obs, action, reward into singular -cycle- obj?

    f += 1

    if done:
        obs = env.reset()
        mario.on_game_end((obs_list, action_list, reward_list))
        obs_list = []
        reward_list = []
        f = 0
        g += 1
env.close()
