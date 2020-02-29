from typing import Type

import retro

from . import Player

class GameMaster():
    """
    Manager class for creating and running Retro games.
    """
    def __init__(self, game, obs_type=retro.Observations.RAM):
        self.env = retro.make(game=game, obs_type=obs_type)

    def run_game(self, player: Type[Player], render=True):
        observation = env.reset()
        frame_count = 0
        while True:
            # action = player.take_action(observation)
            observation, reward, done, info = env.step



class GameResultObject():
    """
    Placeholder for now (del if nothing ends up here)
    Container for results of a run of a game
    """
    pass
