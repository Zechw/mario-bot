import datetime
import numpy as np
import scipy
import tempfile

import uuid
import pdb


storage_base = 'F:'
tmpfile_dir = storage_base + '/gameinputs/'
# db_path = storage_base + '/db/' + datetime.datetime.now() + '.json'

# basic structre exposed to Net users
class BaseNet():
    def __init__(self, shrink_factor=0.5):
        self.shrink_factor = shrink_factor

    def fire(self, observation, debug=None):
        pass
        # Implement this!
        # observation: 2D np.array, downsampled to 112, 120
        # return: 1D arr of 0 or 1 bool, ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

    def on_game_end(self, game):
        pass

    def downscale_observation(self, obs):
        # 3D to 2D
        two_d = np.sum(obs, axis=2)
        #combine pixels
        downsampled = scipy.misc.imresize(two_d, self.shrink_factor) #also avgs back to 0-255
        return downsampled

        ## Debug -- show img
        # from matplotlib import pyplot as plt
        # plt.imshow(downsampled, interpolation='nearest')
        # plt.show()
        # pdb.set_trace()

    ##TODO downscale_observation, append obs-action-reward? or different responsibility?

class GameHistorySave(BaseNet):
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
