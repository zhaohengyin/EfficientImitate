import dmc2gym.wrappers
from game.dmcontrol.env_wrapper import DMCWrapper, DMCLowDimWrapper

def make_dmcontrol(domain_name, task_name, seed, frameskip=4):
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed, height=48, width=48,
                       frame_skip=frameskip, from_pixels=True, visualize_reward=False, channels_first=False)

    return DMCWrapper(env)


def make_dmcontrol_lowdim(domain_name, task_name, seed, frameskip=8):
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed,
                       height=96, width=96, frame_skip=frameskip,
                       from_pixels=False, visualize_reward=False, channels_first=False)

    return DMCLowDimWrapper(env)


