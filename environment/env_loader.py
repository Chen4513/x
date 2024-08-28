import environment.pacman.gym_wrapper
import environment.grid.gym_wrapper
import environment.grid.wrappers
import environment.mujoco.invert_pendulum
import environment.mujoco.reacher
import environment.mujoco.hopper

from functools import partial
from ray.tune import register_env
import gym
from gym.wrappers.time_limit import TimeLimit

# Use this function (with additional arguments if necessary) to additionally add wrappers to environments
def env_maker(config, env_name):
    env = None
    if env_name == "pacman":
        env = environment.pacman.gym_wrapper.GymPacman(config)
    elif env_name == "multi_grid":
        env = environment.grid.gym_wrapper.GymMultiGrid(config)
        env = environment.grid.wrappers.FullyObsWrapper(env)
        env = environment.grid.wrappers.ActionMasking(env)

        # env = environment.grid.wrappers.GymCompatWrapper(env) # This wrapper is only needed if gym version is > 0.26!
        # env = environment.grid.wrappers.DoorUnlockBonus(env)

        if config.get("exploration_bonus", True):
            env = environment.grid.wrappers.ExplorationBonus(env)
            
        # env = environment.grid.wrappers.ActionBonus(env)
    elif env_name == "doubledoor":
        env = environment.grid.gym_wrapper.GymDoubleDoor(config)
        env = environment.grid.wrappers.FullyObsWrapper(env)
        env = environment.grid.wrappers.ActionMasking(env)

    elif env_name == "inverted_pendulum":
        env = environment.mujoco.invert_pendulum.InvertedPendulum(config)
        env = TimeLimit(env, max_episode_steps=config["max_steps"])
    elif env_name == "InvertedPendulum-v2":
        return gym.make('InvrertedPendulum-v2')
    
    elif env_name == "reacher":
        env = environment.mujoco.reacher.Reacher()
        env = TimeLimit(env, max_episode_steps=config["max_steps"])
    elif env_name == "Reacher-v2":
        return gym.make('Reacher-v2')

    elif env_name == "hopper":
        env = environment.mujoco.hopper.Hopper(forward_reward_weight= config["forward_reward_weight"], exclude_current_positions_from_observation = config['exclude_current_positions_from_observation'])
        env = TimeLimit(env, max_episode_steps=config["max_steps"])
    elif env_name == "Hopper-v4":
        return gym.make('Hopper-v4', forward_reward_weight=config["forward_reward_weight"])
    
    else:
        raise("Unknown environment {}".format(env_name))

    return env


def register_envs():
    register_env('Hopper-v4', partial(env_maker, env_name = "Hopper-v4"))
    register_env('Reacher-v2', partial(env_maker, env_name = "Reacher-v2"))
    register_env('InvrertedPendulum-v2', partial(env_maker, env_name = "InvrertedPendulum-v2"))
    register_env("pacman", partial(env_maker, env_name = "pacman"))
    register_env("multi_grid", partial(env_maker, env_name = "multi_grid"))
    register_env("doubledoor", partial(env_maker, env_name = "doubledoor"))
    register_env("inverted_pendulum", partial(env_maker, env_name = "inverted_pendulum"))
    register_env("reacher", partial(env_maker, env_name = "reacher"))
    register_env("hopper", partial(env_maker, env_name = "hopper"))