from types import SimpleNamespace
import numpy as np
from tensorflow import keras
from NetTools import BaseNet

#genetic evolution? NEAT? just a randomizer for now
class LuigiNet(BaseNet):
    def __init__(self, **kw):
        super().__init__(**kw)
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
