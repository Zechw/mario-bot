import retro
import time

from network import MarioNet, LuigiNet
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
        obs = mario.downscale_observation(obs)
        action = mario.fire(obs)

        obs_list.append(obs)
        action_list.append(action)
        if r == 0:
            r = -1
        reward_list.append(r)
        obs, reward, done, info = env.step(action)
                                # ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

        print(g, f, r, action, info)
        r = reward #reset reward
        # pdb.set_trace()
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
