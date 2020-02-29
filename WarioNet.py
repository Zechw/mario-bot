import random
import uuid
import numpy as np
from tensorflow import keras
from NetTools import GameHistorySave_memory

# q implementation on raw memory
class WarioNet(GameHistorySave_memory):
    def __init__(self):
        super().__init__(max_memory_size=10000)
        self.games_to_play_per_training = 1
        self.mini_batch_size = 100
        self.game_i = 0
        self.training_epochs = 2
        self.q_epochs = 2

        self.discount_factor = 0.7
        self.inherent_randomness = 0.01

        self.net = keras.models.Sequential()
        self.net.add(keras.layers.Dense(2048, input_shape=(10240,), activation='sigmoid', kernel_initializer='zeros'))
        self.net.add(keras.layers.Dense(256, activation='sigmoid', kernel_initializer='zeros'))
        self.net.add(keras.layers.Dense(16, activation='sigmoid', kernel_initializer='zeros'))
        self.net.add(keras.layers.Dense(6, kernel_initializer='zeros'))
        self.net.compile(optimizer=keras.optimizers.SGD(0.1),
                        loss='mse',
                        metrics=['accuracy'])

    def fire(self, observation, debug):
        out = self.net.predict(np.array([observation]))[0]
        rand_out = [x + (np.random.normal() * self.inherent_randomness) for x in out]
        print(['%0.3f' % round(x, 3) for x in out], [round(x, 3) for x in rand_out], "\t", debug)
        return self.net_to_controller(rand_out)

    def on_game_end(self, game):
        super().on_game_end(game)
        self.game_i += 1
        if self.game_i % self.games_to_play_per_training == 0:
            self.train()

    def train(self):
        print('--training--')
        game_samples = random.sample(self.game_history, min(self.mini_batch_size, len(self.game_history)))

        for q_i in range(self.q_epochs):
            print('--q:', q_i+1, '/', self.q_epochs)
            inputs = []
            desired_outputs = []
            for game_i, game in enumerate(game_samples):
                observations = game[0]
                actions = game[1]
                rewards = game[2]
                print('--predicting-- game:', game_i, '--', len(observations), '--', sum(rewards))
                actions = [self.controller_to_net(x) for x in actions]
                predictions = self.net.predict(np.array(observations))
                for step_i, observation in enumerate(observations):
                    try:
                        # output of outlook and reward for next state
                        max_q = max(abs(predictions[step_i+1]))
                        reward = rewards[step_i+1] + (self.discount_factor * max_q)
                    except:
                        reward = 0
                                # GAME OVER (TODO: neg reward?)
                    inputs.append(observation)
                    d_out = [((x-0.5) * 2 * reward) for x in actions[step_i]]
                    desired_outputs.append(d_out)

                    # print(reward)
                    # print(actions[step_i])
                    # print(d_out)
            print(np.round(np.min(desired_outputs, axis=0), 2))
            print(np.round(np.max(desired_outputs, axis=0), 2))
            print(np.round(np.sum(desired_outputs, axis=0), 2))
            print(np.round(np.mean(desired_outputs, axis=0), 3))
            # pdb.set_trace()

            print('--fitting--')
            inputs = np.array(inputs)
            desired_outputs = np.array(desired_outputs)
            if desired_outputs.size != 0:
                self.net.fit(inputs, desired_outputs, epochs=self.training_epochs)


    @staticmethod
    def controller_to_net(buttons):
        return [buttons[0]] + buttons[4:]

    @staticmethod
    def net_to_controller(buttons):
        buttons = [1 if x > 0 else 0 for x in buttons] # cast to input bool
        return [buttons[0], 0, 0, 0] + buttons[1:]
