import retro
import time

from network import MarioNet, LuigiNet
mario = LuigiNet()

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
