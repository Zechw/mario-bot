import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
import tempfile
import pdb
import time
from types import SimpleNamespace
import random
import gc

#base implementation
class BaseNet():
    def __init__(self):
        self.game_history = [] # [([obs], [action], [reward]), ...]
        self.max_memory_size = 1000

    ## TODO generate random games?

    def unpack_history(self):
        observation_sets = []
        for game in self.game_history:
            observation_sets.append(self.depickle_path(game[0]))
        return observation_sets

    @staticmethod
    def depickle_path(path):
        with open(path, 'rb') as f:
            return np.load(f)

    def on_game_end(self, game):
        print('--packing--')
        start = time.time()
        with tempfile.NamedTemporaryFile(dir='F:', delete=False) as tmp_file:
            # pdb.set_trace()
            np.save(tmp_file, game[0], allow_pickle=False)
            tmp_file.flush()
            path = tmp_file.name
        print('  runtime:', time.time()-start)
        self.game_history.append( (path, game[1], game[2]) )
        if len(self.game_history) > self.max_memory_size:
            ## TODO delete old gamepickle
            self.game_history.pop(0)

    @staticmethod
    def downscale_observation(obs, shrink_factor=0.5):

        # 3D to 2D
        two_d = np.sum(obs, axis=2)

        #combine pixels
        downsampled = scipy.misc.imresize(two_d, shrink_factor)

        # from matplotlib import pyplot as plt
        # plt.imshow(downsampled, interpolation='nearest')
        # plt.show()
        # pdb.set_trace()

        return downsampled # / (255 * 3)
        # return np.reshape(downsampled, (round(len(obs)*shrink_factor), round(len(obs[0])*shrink_factor), 1))


# q implementation
class MarioNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.training_epochs = 1
        self.q_epochs = 4
        self.discount_factor = 0.8
        self.inherent_randomness = 0.1
        self.net = keras.models.Sequential()
        self.net.add(keras.layers.Conv2D(filters=64,
                                    kernel_size=40,
                                    input_shape=(112, 120, 1), #downsampled b&w 50%
                                    data_format="channels_last",
                                    activation="relu"))
        self.net.add(keras.layers.Dense(32, activation='sigmoid'))
        self.net.add(keras.layers.MaxPooling2D())
        self.net.add(keras.layers.Flatten())
        self.net.add(keras.layers.Dense(256, activation='relu'))
        self.net.add(keras.layers.Dense(32, activation='sigmoid'))
        self.net.add(keras.layers.Dense(9))
        self.net.compile(optimizer=tf.train.AdamOptimizer(0.1),
                        loss='mse',
                        metrics=['accuracy'])

    def fire(self, observation):
        # o = self.downscale_observation(observation)
        o = np.reshape(observation, (len(observation), len(observation[0]), 1))
        out = self.net.predict(np.array([o]))[0]
        print([round(x, 3) for x in out])
        out = [x + (np.random.normal() * self.inherent_randomness) for x in out]
        print([round(x, 3) for x in out])
        return [1 if x > 0 else 0 for x in out]

    def on_game_end(self, game):
        super().on_game_end(game)
        self.train()

    def train(self):
        print('--training--')
        print('--unpacking', len(self.game_history), 'games--')
        start = time.time()
        observation_sets = self.unpack_history()
        print('  runtime:', time.time()-start)

        for q_i in range(self.q_epochs):
            print('--q:', q_i+1, '/', self.q_epochs)
            inputs = []
            desired_outputs = []
            for game_i, game in enumerate(self.game_history):
                observations = observation_sets[game_i]
                observations = [np.reshape(x, (len(x), len(x[0]), 1)) for x in observations]
                actions = game[1]
                rewards = game[2]
                # pdb.set_trace()
                print('--predicting-- game:', game_i, '--', len(observations), '--', sum(rewards))
                start = time.time()
                predictions = self.net.predict(np.array(observations))
                for step_i, observation in enumerate(observations):
                    try:
                        # output of outlook and reward for next state
                        max_q = max(abs(predictions[step_i+1]))
                        reward = rewards[step_i+1] + (self.discount_factor * max_q)
                    except:
                        reward = max(rewards) * -1
                                # GAME OVER
                    inputs.append(observation)
                    desired_outputs.append([((x-0.5) * reward) for x in actions[step_i]])

            print('--fitting--')
            inputs = np.array(inputs)
            desired_outputs = np.array(desired_outputs)
            if desired_outputs.size != 0:
                self.net.fit(inputs, desired_outputs, epochs=self.training_epochs)

#genetic evolution? NEAT? just a randomizer for now
class LuigiNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.population_size = 3
        self.bots = []
        self.init_bots()
        self.set_current_bot(0)

    def init_bots(self):
            for i in range(self.population_size):
                self.create_new_bot()

    def create_new_bot(self):
        bot = SimpleNamespace()
        bot.scores = []
        bot.net = keras.models.Sequential()
        bot.net.add(keras.layers.Conv2D(filters=64,
                                    kernel_size=40,
                                    input_shape=(112, 120, 1), #downsampled b&w 50%
                                    data_format="channels_last",
                                    activation="relu"))
        bot.net.add(keras.layers.Dense(32, activation='sigmoid'))
        bot.net.add(keras.layers.MaxPooling2D())
        bot.net.add(keras.layers.Flatten())
        bot.net.add(keras.layers.Dense(32, activation='sigmoid'))
        bot.net.add(keras.layers.Dense(9))
        self.bots.append(bot)
        # pdb.set_trace()


    def set_current_bot(self, index):
        self.current_bot_index = index
        self.current_bot = self.bots[index]

    def del_current_bot(self):
        del self.bots[self.current_bot_index]

    def fire(self, observation):
        observation = np.reshape(observation, (len(observation), len(observation[0]), 1))
        out = self.current_bot.net.predict(np.array([observation]))[0]
        print([round(x, 3) for x in out])
        return [1 if x > 0 else 0 for x in out]

    def on_game_end(self, game):
        self.current_bot.scores.append(sum(game[2]))
        # super().on_game_end(game) # Dont pack files
        i = self.current_bot_index + 1
        if i >= len(self.bots):
            i = 0
            self.scramble_bots()
        self.set_current_bot(i)

    def scramble_bots(self):
        print('scrambling')
        print([b.scores for b in self.bots])
        avg_score = sum([max(b.scores) for b in self.bots])/len(self.bots)
        print(avg_score)
        k_sess = keras.backend.get_session()
        for bot_i, bot in enumerate(self.bots):
            if max(bot.scores) <= avg_score:
                print('x')
                self.bots[bot_i].scores = []
                for layer_i in range(len(bot.net.layers)):
                    if hasattr(self.bots[bot_i].net.layers[layer_i], 'kernel'):
                        print('.')
                        self.bots[bot_i].net.layers[layer_i].kernel.initializer.run(session=k_sess)
        # pdb.set_trace()
        # print('--')
        while len(self.bots) < self.population_size:
            self.create_new_bot()
