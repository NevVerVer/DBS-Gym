import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from typing import Any, Callable, Optional, Union
from scipy.signal import filtfilt
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecEnv, VecMonitor,
    is_vecenv_wrapped,
    )
# We import class and utils!
from environment.utils import generate_w0_with_locus, band_pass_envelope
from environment.env import SpatialKuramoto

# We import env version 
from environment.env_configs.env2 import (
    n_neurons, grid_size, coord_modif,
    eval_envs_list, checking
)
np.random.seed(228)


def make_env(d):
    """
    Creates environment for eval
    """
    # env = Monitor(SpatialKuramoto(params_dict=d),
    #                   filename=None)
    env = SpatialKuramoto(params_dict=d)
    return env


def evaluate_policy_(
    model,
    env: Union[gym.Env, VecEnv],
    n_envs,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Mine version of evaluate_policy()
    """

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    true_lfp = [[] for n in range(n_envs)]
    actions_list = []
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)

        for n in range(n_envs):
            true_lfp[n].append(env.get_attr('theta_mean')[n])  

        actions_list.append([u[0] for u in actions])
        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if dones[i]:
                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

    true_lfp_ = []
    for env_lfp in true_lfp:
        true_lfp_.append(np.concatenate(env_lfp))

    true_lfp_ = np.asarray(true_lfp_)
    actions_list = np.asarray(actions_list)
    return episode_rewards, true_lfp_, actions_list


def calc_psd_for_simple_eval(sig_envs, psd_dt,
                             cut_idx=1500, log_scale=True,
                             beta_a=12.5, beta_b=21):
    bbpow_list = []
    for ii, sig in enumerate(sig_envs):
        sig_filt, _ = band_pass_envelope(sig, 1/psd_dt, order=2)
        ft = np.abs(np.fft.rfft(sig_filt)/sig_filt.shape[0])**2 * 2
        freq = np.fft.rfftfreq(sig_filt.shape[0], psd_dt)
        ft = filtfilt([1,1,1,1,1,1,1,1,1,1,1,1,], 5, ft)
        # Also calculate beta band power
        idx = np.where((freq > beta_a) & (freq < beta_b))
        bbpow_list.append(np.sum(ft[idx]))

    return np.asarray(bbpow_list)


def evaluate_hf_dbs(model, eval_env: list, n_eval_episodes,
                    render=False, deterministic=True,
                    warn=False, callbacks_=None):
    reward_list = []
    bbpows_list, u_energy_list = [], []

    for env in eval_env:
        episode_rewards, true_lfp, actions_list = evaluate_policy_(
                    model, env,
                    n_envs=1,
                    n_eval_episodes=n_eval_episodes,
                    render=render,
                    deterministic=deterministic,
                    return_episode_rewards=True,
                    warn=warn, callback=callbacks_,)

        reward_list.append(episode_rewards[0])

        bbpow = calc_psd_for_simple_eval(true_lfp, psd_dt=0.0005)  # NOTE: hardcode dt
        bbpows_list.append(bbpow[0])

        u_energy = np.sum(np.abs(actions_list), axis=0)
        u_energy_list.append(u_energy)
        
    bbpows_list = np.asarray(bbpows_list)
    u_energy_list = np.asarray(u_energy_list)
    reward_list = np.asarray(reward_list)

    mean_reward, std_reward = np.mean(reward_list), np.std(reward_list, ddof=1)
    bbpow_mean, bbpow_sd = np.mean(bbpows_list), np.std(bbpows_list, ddof=1)
    e_mean, e_sd = np.mean(u_energy_list), np.std(u_energy_list, ddof=1)

    print(f'Reward mean={mean_reward}, std={std_reward}')
    print(f'BBpow mean={bbpow_mean}, std={bbpow_sd}')
    print(f'Energy mean={e_mean}, std={e_sd}')

    return bbpow_mean, bbpow_sd, e_mean, e_sd

class HFDBS():
    def __init__(self, action: float):
        """
        A naive agent that always returns the same action.
        """
        self.action = action
    def predict(self, observation, state=None,
                episode_start=None, deterministic=True):
        return [[self.action]], None 


if __name__ == "__main__":

    # Set parameters for validation 
    NUMBER_OF_ENV = 5
    each_env_run_episodes = 25

    # 1. Instantiate environments for evaluation
    eval_envs_list_new = []
    for n_env in range(NUMBER_OF_ENV):
        eval_d = eval_envs_list[n_env]

        (w0_eval, ncoords, ngrid,
            w0_temp_eval, w_locus_eval, lmask_eval) = generate_w0_with_locus(
                    n_neurons, grid_size,
                    coord_modif,
                    locus_center=eval_d['locus_center'],
                    locus_size=eval_d['locus_size'],
                    wmuL=17, wsdL=1, 
                    show=False, vertical_layer=4)
        
        eval_d['reward_func'] = 'bbpow_action'
        eval_d['neur_coords'] = ncoords
        eval_d['neur_grid'] = ngrid

        eval_d['w0'] = w0_eval
        eval_d['w0_without_locus'] = w0_temp_eval
        eval_d['locus_without_w0'] = w_locus_eval
        eval_d['locus_mask'] = lmask_eval

        eval_d['dbs_action_bounds'] = [-5, 5]    # NOTE: IMPORTANT!!!!
        
        eval_envs_list_new.append(make_env(eval_d))
    envs_cpu = eval_envs_list_new

    # 2. Set the model
    model = HFDBS(action=1.)   # we rescale action inside the step() method

    # 3. Print all info to not mistake
    print('^^'*30)
    print('THE ENVIROMENT IS:', checking)
    print('NUMBER OF ENVS:', NUMBER_OF_ENV)
    print('Each env with run for: ', each_env_run_episodes, ' times')

    print('MODEL PREDICT: ', model.predict(np.ones(5)))
    print('SCALE TO INSIDE env', envs_cpu[0].params_dict['dbs_action_bounds'])
    print('Episode len:', envs_cpu[0].params_dict['total_episode_len'])

    print('Temporal drift:', envs_cpu[0].params_dict['temporal_drift'])
    print('Spatial features:', envs_cpu[0].params_dict['spatial_feature'])
    print('^^'*30)
    r = envs_cpu[0].params_dict['dbs_action_bounds'][1]
    print('Energy is = ',
          f'e * {r} / {each_env_run_episodes}')

    # 4. Run evaluation 
    bbpow_mean, bbpow_sd, e_mean, e_sd = evaluate_hf_dbs(model, envs_cpu,
                    
                    # NOTE: IMPORTANT: EACH environment run this num of episodes
                    n_eval_episodes=each_env_run_episodes, 

                    render=False, deterministic=True,
                    warn=False, callbacks_=None)
    print('Energy is = ', e_mean * envs_cpu[0].params_dict['dbs_action_bounds'][1] / each_env_run_episodes)