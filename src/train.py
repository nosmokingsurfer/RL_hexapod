#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym, roboschool
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import imageio
import shutil
from multiprocessing.pool import ThreadPool


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def init_gyms(env_name, env_count):
    env, obs_dim, act_dim = init_gym(env_name)
    env.close()
    return [init_gym(env_name)[0] for i in range(env_count)], obs_dim, act_dim


def run_episode(env, policy, scaler, animate=False, logger=None, anim_name='ant_train'):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    rendered_frames = []
    while not done:
        if animate:
            rendered_frames.append(env.render("rgb_array"))
            if int(step * 1000) % 50 == 0:
                print("Rendering: {}%".format(int(step * 100)))
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
        actions.append(action)
        # тут магия, раньше было просто action, но это массив в массиве (зачем?) поэтому action[0]
        obs, reward, done, _ = env.step(action[0])
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    if animate:
        p = '~/Desktop'
        if logger is not None:
            p = logger.path
        imageio.mimwrite('{}/{}.mp4'.format(p, anim_name), np.array(rendered_frames), fps=60)

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes, animate=False, anim_name='ant_train'):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        if animate:
            print("animating episode {}/{}".format(e+1, episodes))
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler, animate=animate, logger=logger, anim_name=anim_name)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def run_episode_wrapper(env, policy, scaler, animate, logger, anim_name):
    observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler, animate=animate, logger=logger,
                                                           anim_name=anim_name)
    trajectory = {'observes': observes,
                  'actions': actions,
                  'rewards': rewards,
                  'unscaled_obs': unscaled_obs}
    return trajectory


def run_policy_parallel(env_list, policy, scaler, logger, episodes, animate=False, anim_name='ant_train', thread_num=4):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env_list: ai gym environments, 1 per each episode
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    assert (len(env_list) == episodes)

    params = [(env, policy, scaler, False, None, '' + str(num)) for num, env in enumerate(env_list)]
    pool = ThreadPool(thread_num)
    trajectories = pool.starmap(run_episode_wrapper, params)
    pool.close()
    pool.join()
    total_steps = sum([t['observes'].shape[0] for t in trajectories])
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode, train_time, sim_time, policy_time, val_time):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_full_train_time': str(train_time).split('.')[0],
                '_time_simulation': str(sim_time).split('.')[0],
                '_time_policy_train': str(policy_time).split('.')[0],
                '_time_value_train': str(val_time).split('.')[0]
                })


def estimate_time_left(episode, num_episodes, train_time):
    full_seconds = (train_time.total_seconds() * (num_episodes / episode)) - train_time.total_seconds()
    hours, remainder = divmod(full_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))


def log_train_info(logger, num_episodes, start_time_str, gait_name, gait_length, batch_size, restore_path):
    if num_episodes > 0:
        # log train info
        with open(logger.path + '/train_info.txt', 'w+') as finfo:
            finfo.write("\tTrain info:\n")
            finfo.write("Date:    \t\t{}\n".format(start_time_str))
            finfo.write("Epizodes:\t\t{}\n".format(-1))
            finfo.write("gait_name:\t\t{}\n".format(gait_name))
            if gait_name is not None:
                finfo.write("gait_length:\t\t{}\n".format(gait_length))
            finfo.write("batch_size:\t\t{}\n".format(batch_size))
            finfo.write("Restore path:\t\t{}\n\n".format(restore_path))
            finfo.write("Out path:\t\t{}\n".format(logger.path))

    else:
        # copy info if inference
        open_mode = 'a'
        try:
            shutil.copy(restore_path + '/train_info.txt', logger.path + '/train_info.txt')
        except Exception as e:
            print(e)
            open_mode = 'w+'
        with open(logger.path + '/train_info.txt', open_mode) as finfo:
            finfo.write("\n\tRendering inference:\n")
            finfo.write("Date:    \t\t{}\n".format(start_time_str))
            finfo.write("Restore path:\t\t{}\n".format(restore_path))
            finfo.write("Out path:\t\t{}\n\n".format(logger.path))



def update_train_info(logger, num_episodes):
    lines = None
    with open(logger.path + '/train_info.txt', 'r') as finfo:
        lines = finfo.readlines()
        lines[2] = "Epizodes:\t\t{}\n".format(num_episodes)
    with open(logger.path + '/train_info.txt', 'w') as finfo:
        finfo.writelines(lines)



def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, restore_path, out_path, thread_count, animation_mode, gait_name, gait_length, gaits_config_path):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """
    killer = GracefulKiller()
    # restore_path = os.path.abspath(restore_path)
    env, obs_dim, act_dim = init_gym(env_name)
    log_rewards = (num_episodes == 0)
    env_list = []
    if thread_count > 1:
        env_list, obs_dim, act_dim = init_gyms(env_name, batch_size)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    start_time = datetime.now()  # create unique directories
    start_time_str = start_time.strftime("%b-%d/%H.%M.%S")
    logger = Logger(logname=env_name, now=start_time_str, out_path=out_path)
    env.env.set_params(gaits_config_path=gaits_config_path, gait_name=gait_name, gait_cycle_len=gait_length, out_path=logger.path, log_rewards=log_rewards, render_mode=animation_mode)
    scaler = Scaler(obs_dim)

    val_func = NNValueFunction(obs_dim, logger, restore_path)
    policy = Policy(obs_dim, act_dim, kl_targ, logger, restore_path)

    log_train_info(logger, num_episodes, start_time_str, gait_name, gait_length, batch_size, restore_path)

    # run a few episodes of untrained policy to initialize scaler:
    episode = 0
    try:
        if restore_path is None:
            print("\nInitializing scaler (may take some time)... ")
            run_policy(env, policy, scaler, logger, episodes=5)
            print("Done\n")
        else:
            scaler.load(restore_path, obs_dim)

        while episode < num_episodes:
            sim_time = datetime.now()
            if thread_count > 1:
                trajectories = run_policy_parallel(env_list, policy, scaler, logger, episodes=batch_size, thread_num=thread_count)
            else:
                trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
            sim_time = datetime.now() - sim_time

            episode += len(trajectories)
            add_value(trajectories, val_func)  # add estimated values to episodes
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            add_gae(trajectories, gamma, lam)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
            # add various stats to training log:
            train_time = datetime.now() - start_time
            policy_time = datetime.now()
            policy.update(observes, actions, advantages, logger)  # update policy
            policy_time = datetime.now() - policy_time
            val_time = datetime.now()
            val_func.fit(observes, disc_sum_rew, logger)  # update value function
            val_time = datetime.now() - val_time

            log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode, train_time, sim_time, policy_time, val_time)
            logger.write(display=True)  # write logger results to file and stdout
            print("Estimated time left: {}\n".format(estimate_time_left(episode, num_episodes, train_time)))

            if episode % 1000 == 0:
                policy.save()
                val_func.save()
                scaler.save(logger.path)
                print("Data saved at {}\n".format(logger.path))
                update_train_info(logger, episode)
                if animation_mode > 0:
                    run_policy(env, policy, scaler, logger, episodes=1, animate=True, anim_name='epizode_{}'.format(episode))
            if killer.kill_now:
                # if input('Terminate training (y/[n])? ') == 'y':
                #     break
                # killer.kill_now = False
                break
    finally:
        if animation_mode > 0:
            print("Rendering result video")
            try:
                trajectories = run_policy(env, policy, scaler, logger, episodes=1, animate=True, anim_name='final_epizode_{}'.format(episode))
                # for walk analysis
                for t in trajectories:
                    logger.log_trajectory(t)
            except Exception as e:
                print("Failed to animate results, error: {}".format(e))
                raise e

        scaler.save(logger.path)
        logger.close()
        policy.close_sess()
        val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-r', '--restore_path', type=str,
                        help='Restore from saved session and continue training',
                        default=None)
    parser.add_argument('-o', '--out_path', type=str,
                        help='Path for all out data, by default using current working dir.',
                        default=None)
    parser.add_argument('-t', '--thread_count', type=int,
                        help='Number of threads',
                        default=1)
    parser.add_argument('-a', '--animation_mode', type=int, help='Animation mode, 0 - no animation, 1 - side, 2 - top + side + graph',
                        default=0)
    parser.add_argument('-gcp', '--gaits_config_path', type=str,
                        help='Path for directory containing gait.csv',
                        default='./walk_analyse')
    parser.add_argument('-gn', '--gait_name', type=str,
                        help='Gait to train. Default training without any gait',
                        default=None)
    parser.add_argument('-gl', '--gait_length', type=int,
                        help='Gait cycle length',
                        default=30)

    args = parser.parse_args()
    main(**vars(args))
