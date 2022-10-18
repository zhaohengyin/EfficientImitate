import time

from game.dmcontrol import make_dmcontrol_lowdim


env = make_dmcontrol_lowdim('cartpole', 'swingup', 0)
obs = env.reset()
print(obs)

time.sleep(100)

