import retro
import time

import random

from WarioNet import WarioNet
mario = WarioNet()

env = retro.make(game='SuperMarioBros-Nes', obs_type=retro.Observations.RAM)
obs = env.reset()
info = None
obs_list = []
action_list = []
reward_list = []
f = 0
g = 0
r = 0

from NetTools import RandomBot
# mario = RandomBot()

render_inp = input('would you like to render? y/n ')
render = True if render_inp == 'y' or render_inp == '' else False

while True:
    if f % 10 != 0:
        #only play every n frames
        obs, reward, done, _ = env.step(action)
        r += reward
    else:
        # obs = mario.downscale_observation(obs)
        # import pdb; pdb.set_trace()
        action = mario.fire(obs, [g, f, r, info])
        # import pdb; pdb.set_trace()

        obs_list.append(obs)
        action_list.append(action)
        # if r == 0:
        #     r = -1
        reward_list.append(r)
        obs, reward, done, info = env.step(action)
                                # ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
        r = reward #reset reward
        # pdb.set_trace()
        if render:
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




import sys
#    0    1      2        3       4       5       6      7       8
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
def i_c(input, on='#', off='.'):
    return on if input is 1 else off

def render_controller(buttons):
    print(' ' + i_c(buttons[4]))
    print(i_c(buttons[6]) + '+' + i_c(buttons[7]) + i_c(buttons[2], '=', '-') + i_c(buttons[3], '=', '-') + i_c(buttons[0]) + i_c(buttons[8]))
    print(' ' + i_c(buttons[5]))
