import datetime
import numpy as np

from PIL import Image

import tempfile

import uuid
import pdb


storage_base = 'F:'
tmpfile_dir = storage_base + '/gameinputs/'
# db_path = storage_base + '/db/' + datetime.datetime.now() + '.json'

# basic structre exposed to Net users
class BaseNet():
    def fire(self, observation, debug=None):
        raise NotImplementedError

    def on_game_end(self, game):
        pass

class ImageNet(BaseNet):
    def __init__(self, shrink_factor=0.5):
        self.shrink_factor = shrink_factor

    def fire(self, observation, debug=None):
        pass
        # Implement this!
        # observation: 2D np.array, downsampled to 112, 120
        # return: 1D arr of 0 or 1 bool, ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

    def downscale_observation(self, obs):
        # 3D to 2D
        two_d = np.mean(obs, axis=2)


        #combine pixels
        #import scipy
        #downsampled = scipy.misc.imresize(two_d, self.shrink_factor)

        downsampled = Image.fromarray(two_d).resize((120, 112))

        # return downsampled
        ## Debug -- show img

        # from matplotlib import pyplot as plt
        # plt.imshow(downsampled, interpolation='nearest')
        # plt.show()
        # pdb.set_trace()
        return np.array(downsampled)

    ##TODO downscale_observation, append obs-action-reward? or different responsibility?

class GameHistorySave_memory(BaseNet):
    def __init__(self, max_memory_size=1000, **kw):
        super().__init__(**kw)
        self.game_history = [] # [([obs], [action], [reward]), ...] #todo make this it's own, better object
        self.max_memory_size = max_memory_size

    def on_game_end(self, game):
        super().on_game_end(game)
        self.game_history.append( (game[0], game[1], game[2]) )
        if len(self.game_history) > self.max_memory_size:
            self.game_history.pop(0)


class GameHistorySave_disk(BaseNet):
    def __init__(self, max_memory_size=1000, **kw):
        super().__init__(**kw)
        self.game_history = [] # [([obs], [action], [reward]), ...] #todo make this it's own, better object
        self.max_memory_size = max_memory_size




        ########################

                ## TODO HERE: rewrite GHS to save to db

        ########################


    def on_game_end(self, game):
        super().on_game_end(game)
        path = self.save(game[0])
        self.game_history.append( (path, game[1], game[2]) )
        if len(self.game_history) > self.max_memory_size:
            ## TODO delete old gamefile, also store score
            self.game_history.pop(0)

    @classmethod
    def unpack_history(self, games):
        observation_sets = []
        for game in games:
            observation_sets.append(self.load(game[0]))
        return observation_sets

    @staticmethod
    def save(object):
        with tempfile.NamedTemporaryFile(dir=tmpfile_dir, delete=False) as tmp_file:
            np.save(tmp_file, object, allow_pickle=False)
            tmp_file.flush()
        return tmp_file.name


    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return np.load(f)

class NetworkSave(BaseNet):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.network_id = uuid.uuid1().hex







import random
class RandomBot(BaseNet):
    def __init__(self, frame_hold=10, random_hold=True):
        self.games = 0
        self.frame_hold = frame_hold
        self.random_hold = random_hold
        self.frame_counter = 0
        self.reset_mask()

    def reset_mask(self):
        if self.random_hold == True:
            self.frame_hold = random.randint(1,100)
        self.mask = [random.randint(0,1),
            random.randint(0,1),
            random.randint(0,1),
            random.randint(0,1),
            random.randint(0,1),
            random.randint(0,1),
            random.randint(0,1),
            random.randint(0,1),
            random.randint(0,1)]

    # def downscale_observation(self, obs):
    #     return obs

    def fire(self, obs=None, x=None):
        self.frame_counter += 1
        if self.frame_counter % self.frame_hold == 0:
            self.reset_mask()
        return self.mask

    def on_game_end(self, x=None):
        self.frame_counter = 0
        self.games += 1
        print(self.games)
