import retro
import random
import time

env = retro.make(game='SuperMarioBros-Nes')
obs = env.reset()



jump_frames = [
                #    ~30 for max jump height

    (100, 2),
    (128, 2),
    (170, 6),
    (225, 15),
    (260, 20),
    (310, 30),
    (403, 8),
    (446, 0),
    (497, 2),
    (555, 0),
    (598, 2),
    (644, 2),
    (671, 2),
    (694, 2),
    (716, 2),
    (769, 0),
    (805, 0),
    (852, 16),
    (879, 0),
    (942, 16),
    (969, 20),
    (1032, 6),
    (1082, 2),
    (1104, 0),
    (1130, 8),
    (1155, 25),
    (1192, 1),
    (1220, 100)

]

frame = 0
jump_i = 0
next_jump = jump_frames[jump_i]
while True:
    if next_jump is None:
        j = 0
    else:
        j = 1 if frame >= next_jump[0] else 0
        if frame >= next_jump[0] + next_jump[1]:
            jump_i += 1
            try:
                next_jump = jump_frames[jump_i]
            except:
                next_jump = None

    obs, rew, done, info = env.step([1,0,0,0,0,0,0,1,j])
                                   # b ? ? ? ? ? l r a
    env.render()
    print(frame)
    frame += 1

    time.sleep(0.02)
    # if frame > 1190:
    #     input()

    if done:
        obs = env.reset()
        frame = 0
env.close()
