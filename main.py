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
        self.training_epochs = 2
        self.q_epochs = 2
        self.max_memory_size = 32
        self.discount_factor = 0.8
        self.inherent_randomness = 1
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Conv2D(filters=64,
                                    kernel_size=40,
                                    input_shape=(224, 240, 3),
                                    data_format="channels_last",
                                    activation="relu",
                                    kernel_initializer='zeros'))
        self.net.add(keras.layers.MaxPooling2D())
        self.net.add(keras.layers.Flatten())
        self.net.add(keras.layers.Dense(64, activation='sigmoid', kernel_initializer='zeros'))
        self.net.add(keras.layers.Dense(32, activation='sigmoid', kernel_initializer='zeros'))
        self.net.add(keras.layers.Dense(9, kernel_initializer='zeros'))
        self.net.compile(optimizer=tf.train.AdamOptimizer(0.1),
                        loss='mse',
                        metrics=['accuracy'])

    def fire(self, observation):
        out = self.net.predict(np.array([observation]))[0]
        print([round(x, 3) for x in out])
        out = [x + (np.random.normal() * self.inherent_randomness) for x in out]
        print([round(x, 3) for x in out])
        return [1 if x > 0 else 0 for x in out]

    def train(self):
        print('--training--')
        print('--unpacking', len(self.game_history), 'games--')
        start = time.time()
        observation_sets = []
        for game in self.game_history:
            observation_sets.append(self.depickle_path(game[0]))
        print('  runtime:', time.time()-start)

        for q_i in range(self.q_epochs):
            print('--q:', q_i+1, '/', self.q_epochs)
            inputs = []
            desired_outputs = []
            for game_i, game in enumerate(self.game_history):
                observations = observation_sets[game_i]
                actions = game[1]
                rewards = game[2]
                # pdb.set_trace()
                print('--predicting-- game:', game_i, '--', len(observations))
                start = time.time()
                predictions = self.net.predict(np.array(observations))
                for step_i, observation in enumerate(observations):
                    try:
                        # output of outlook and reward for next state
                        max_q = max(abs(predictions[step_i+1]))
                        reward = rewards[step_i+1] + (self.discount_factor * max_q)
                    except:
                        reward = sum(rewards) * -1
                                # GAME OVER
                    inputs.append(observation)
                    desired_outputs.append([((x-0.5) * reward) for x in actions[step_i]])

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
        print('--packing--')
        start = time.time()
        with tempfile.NamedTemporaryFile(dir='F:', delete=False) as tmp_file:
            # pdb.set_trace()
            pickle.dump(game[0], tmp_file)
            tmp_file.flush()
            path = tmp_file.name
        print('  runtime:', time.time()-start)
        self.game_history.append( (path, game[1], game[2]) )
        if len(self.game_history) > self.max_memory_size:
            ## TODO delete old gamepickle
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
r = 0
while True:
    if f % 10 != 0:
        #only play every n frames
        obs, reward, done, _ = env.step(action)
        r += reward
    else:
        action = mario.fire(obs)

        obs_list.append(obs)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
                                # [b ? ? ? ? ? l r a]

        print(g, f, r, action, info)
        reward_list.append(r)
        r = reward #reset reward
        # pdb.set_trace()


    # bundle obs, action, reward into singular -cycle- obj?

    env.render()
    f += 1

    if done:
        obs = env.reset()
        mario.on_game_end((obs_list, action_list, reward_list))
        obs_list = []
        reward_list = []
        f = 0
        g += 1
env.close()
