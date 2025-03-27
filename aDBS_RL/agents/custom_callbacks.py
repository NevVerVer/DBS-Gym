import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
from neurokuramoto.utils import band_pass_envelope, units2sec, calc_envelope
import gym 
import warnings
from typing import Any, Callable, Optional, Union
from scipy.signal import filtfilt

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecEnv, VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization)
import time


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, csv_name, verbose=0):
        super().__init__(verbose=verbose)
        self.csv_name = csv_name

    def log_main_metrics(self, n, m, x):
        self.logger.record(f"{n}/{m}/mean", np.mean(x))
        self.logger.record(f"{n}/{m}/std", np.std(x, ddof=1))
        self.logger.record(f"{n}/{m}/cum", np.sum(x))
    
    def log_to_csv(self, x, csv_name):
        stats = {'A': x[:3], 'B': x[3:6], 'C': x[6:9]}
        df = pd.DataFrame(data=stats)
        df.to_csv(csv_name, mode='a', index=False, header=None) 

    def log_psd(self, sig):
        sig_filt, _ = band_pass_envelope(sig, 1/self.psd_dt, order=2)
        ft = np.abs(np.fft.rfft(sig_filt)/sig_filt.shape[0])**2 * 2
        freq = np.fft.rfftfreq(sig_filt.shape[0], self.psd_dt)
        ft = filtfilt([1,1,1,1,1,1,1,1,1,1,1,1,], 5, ft)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cut_idx = 2000
        log_scale = True
        if log_scale:
            ax.semilogy(freq[:cut_idx], ft[:cut_idx], label='with DBS')
            ax.semilogy(self.freq_R[:cut_idx], self.ft_R[:cut_idx], label='noDBS')
            ax.set_ylim([10e-8, 10-2])
        else:
            ax.plot(freq[:cut_idx], ft[:cut_idx])
        ax.set_xlabel('Hz')
        ax.set_ylabel('Volt**2/hz')
        # beta ranges: low (13–20 Hz) and high (21–35 Hz)
        ax.axvspan(4, 12.5, alpha=0.15, color='green', label='LF')
        ax.axvspan(12.5, 21, alpha=0.15, color='blue', label='Low beta band')
        ax.axvspan(21, 33.5, alpha=0.15, color='red', label='High beta band')
        ax.legend()
        ax.set_title(f'Power Spectral Density, beta-filt, log_scale')
        self.logger.record("figures/psd", Figure(fig, close=True),
                            exclude=("stdout", "log", "json", "csv"))
        plt.close()
        # Also calculate beta band power
        idx = np.where((freq > self.beta_a) & (freq < self.beta_b))
        band_power = np.sum(ft[idx])
        return band_power

    def log_lfp_env(self, sig, current_time):
        n = sig.shape[0]
        nn = self.reference_lfp.shape[0]
        if nn >= n:
            nn = n
        sig_filt, _ = band_pass_envelope(sig, 1/self.psd_dt, order=2)
        times = np.arange(current_time-self.psd_dt*n, current_time, self.psd_dt)[:n]

        fig, ax = plt.subplots(1, 1, figsize=(14, 3), dpi=150)
        ax.plot(times[:nn], self.reference_lfp[:nn], label='LFP no DBS', alpha=0.3)
        ax.plot(times[:nn], sig_filt[:nn], label='LFP with DBS')
        ax.axhline(0, linestyle='--', c='grey',)
        ax.set_title(f'LFP Amplitude, beta-filt')
        ax.set_xlabel('Time, [counts]')
        ax.set_ylabel('Amplitude, [unit V]')
        ax.legend(loc='lower left')
        ax.set_ylim([-0.35, 0.35])
        self.logger.record("figures/lfp", Figure(fig, close=True),
                            exclude=("stdout", "log", "json", "csv"))
        plt.close()

    def log_hist(self, sig):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.hist(sig, bins=self.hist_bins)
        ax.set_ylim([0., 170.])
        hist_, _ = np.histogram(sig, bins=self.hist_bins)
        ax.set_title(f'Actions hist, Nbins={self.hist_bins}, [{hist_[0]}, {hist_[-1]}]')
        self.logger.record("figures/uhist", Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv"))
        plt.close()

    def _on_training_start(self):
        self._log_freq = 50
        # hyperparams for logging
        self.log_scale = True
        self.beta_a, self.beta_b = 12.5, 33.5
        self.hist_bins = 90  # for action
        self.psd_dt = units2sec(self.training_env.get_attr('params_dict')[0]['verbose_dt'])
        self.time_sec = units2sec(self.training_env.get_attr('params_dict')[0]['transient_state_len'])

        self.step_len_sec = units2sec(self.training_env.get_attr('params_dict')[0]['electrode_width']) +\
                            units2sec(self.training_env.get_attr('params_dict')[0]['electrode_pause'])
        # for plotting
        self.reference_lfp = np.load('data/reference_fileterd_lfp.npy')
        sig_filt_R, _ = band_pass_envelope(self.reference_lfp, 1/self.psd_dt, order=2)
        self.ft_R = np.abs(np.fft.rfft(sig_filt_R)/sig_filt_R.shape[0])**2 * 2
        self.freq_R = np.fft.rfftfreq(sig_filt_R.shape[0], self.psd_dt)
        self.ft_R = filtfilt([1,1,1,1,1,1,1,1,1,1,1,1,], 5, self.ft_R)

        # lists for logging 
        self.per_ep_reward = []
        self.per_ep_action = []
        self.per_ep_lfp = []

    def _on_step(self) -> bool:
        """
        Called it at each step of env
        NOTE: Be careful with reset() and step() shared variables.
        at the last rollout step - reset() is called.
        """
        self.per_ep_reward.append(self.locals['rewards'][0])
        self.per_ep_action.append(self.training_env.get_attr('u')[0])

        self.per_ep_lfp.append(self.training_env.get_attr('theta_mean')[0])
        self.time_sec += self.step_len_sec

        ### logging at the episode end 
        if True in self.locals["dones"]:
            # log reward
            self.log_main_metrics('per_episode', 'Reward', np.asarray(self.per_ep_reward))
            self.log_main_metrics('per_episode', 'action', np.asarray(self.per_ep_action))
            self.logger.record("per_episode/action/energy",
                            np.sum(np.abs(np.asarray(self.per_ep_action))))
            self.log_hist(np.asarray(self.per_ep_action))  # log actions hist

            # log LFP amp
            _lfp_ep = np.concatenate(self.per_ep_lfp)
            self.log_main_metrics('per_episode', 'envelope', calc_envelope(_lfp_ep))
            self.log_lfp_env(_lfp_ep, self.time_sec)  # log lfp plot

            # log beta band power and PSD
            bbpow = self.log_psd(_lfp_ep)
            self.logger.record(f"per_episode/bbpow", bbpow)

            self.logger.dump(self.num_timesteps)
            self.per_ep_reward = []
            self.per_ep_lfp = []
            self.per_ep_action = []
            self.time_sec = units2sec(self.training_env.get_attr('params_dict')[0]['transient_state_len'])
        return True
    
    def _on_training_end(self) -> None:
        pass


def evaluate_policy_(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
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

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    n_envs = env.num_envs
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
        
        ###=====================================
        for n in range(n_envs):
            true_lfp[n].append(env.get_attr('theta_mean')[n])  

        actions_list.append([u[0] for u in actions])
        ###=====================================
        
        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

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


class EvalCallback_(EventCallback):
    """
    Mine version of EvalCallback
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        self.psd_dt = units2sec(eval_env.get_attr('params_dict')[0]['verbose_dt'])
        self.beta_a, self.beta_b = 12.5, 33.5
        self.hist_bins = 90  # for action
        self.log_env_num = 0  # we will save actual signal only from 1 env

        # for plotting
        self.reference_lfp = np.load('data/reference_fileterd_lfp.npy')
        sig_filt_R, _ = band_pass_envelope(self.reference_lfp, 1/self.psd_dt, order=2)
        self.ft_R = np.abs(np.fft.rfft(sig_filt_R)/sig_filt_R.shape[0])**2 * 2
        self.freq_R = np.fft.rfftfreq(sig_filt_R.shape[0], self.psd_dt)
        self.ft_R = filtfilt([1,1,1,1,1,1,1,1,1,1,1,1,], 5, self.ft_R)

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations.npy")
        self.log_path = log_path
        self.evaluations_results: list[list[float]] = []
        self.evaluations_timesteps: list[int] = []
        self.eval_actions = []
        self.eval_lfp = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    # =================================================================================
    def log_psd_for_envs(self, sig_envs, cut_idx=1900, log_scale=True):
        bbpow_list = []
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for ii, sig in enumerate(sig_envs):
            sig_filt, _ = band_pass_envelope(sig, 1/self.psd_dt, order=2)
            ft = np.abs(np.fft.rfft(sig_filt)/sig_filt.shape[0])**2 * 2
            freq = np.fft.rfftfreq(sig_filt.shape[0], self.psd_dt)
            ft = filtfilt([1,1,1,1,1,1,1,1,1,1,1,1,], 5, ft)
            if log_scale:
                ax.semilogy(freq[:cut_idx], ft[:cut_idx], label=f'with DBS, {ii}',
                            alpha=0.6)
                ax.set_ylim([10e-8, 10-2])
            else:
                ax.plot(freq[:cut_idx], ft[:cut_idx])

            # Also calculate beta band power
            idx = np.where((freq > self.beta_a) & (freq < self.beta_b))
            bbpow_list.append(np.sum(ft[idx]))

        ax.semilogy(self.freq_R[:cut_idx], self.ft_R[:cut_idx], label='noDBS')
        ax.set_xlabel('Hz')
        ax.set_ylabel('Volt**2/hz')
        # beta ranges: low (13–20 Hz) and high (21–35 Hz)
        ax.axvspan(4, 12.5, alpha=0.15, color='green', label='LF')
        ax.axvspan(12.5, 21, alpha=0.15, color='blue', label='Low beta band')
        ax.axvspan(21, 33.5, alpha=0.15, color='red', label='High beta band')
        ax.legend()
        ax.set_title(f'Power Spectral Density, beta-filt, log_scale')
        self.logger.record("eval/figures/psd", Figure(fig, close=True),
                            exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return np.asarray(bbpow_list)


    def log_lfp_env(self, sig, current_time):
        n = sig.shape[0]
        nn = self.reference_lfp.shape[0]
        if nn >= n:
            nn = n
        sig_filt, _ = band_pass_envelope(sig, 1/self.psd_dt, order=2)
        times = np.arange(current_time-self.psd_dt*n, current_time, self.psd_dt)[:n]

        fig, ax = plt.subplots(1, 1, figsize=(14, 3), dpi=150)
        ax.plot(times[:nn], self.reference_lfp[:nn], label='LFP no DBS', alpha=0.3)
        ax.plot(times, sig_filt, label='LFP with DBS')
        ax.axhline(0, linestyle='--', c='grey',)
        ax.set_title(f'LFP Amplitude, beta-filt')
        ax.set_xlabel('Time, [counts]')
        ax.set_ylabel('Amplitude, [unit V]')
        ax.legend(loc='lower left')
        ax.set_ylim([-0.35, 0.35])
        self.logger.record("eval/figures/lfp", Figure(fig, close=True),
                            exclude=("stdout", "log", "json", "csv"))
        plt.close()


    def log_hist_for_envs(self, sig_envs):

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for i, sig in enumerate(sig_envs):
            ax.hist(sig, bins=self.hist_bins, alpha=0.3)
            if i == 2:   # TMP NOTE
                break
        # ax.set_ylim([0., 10.])  # TMP NOTE
        hist_, _ = np.histogram(sig, bins=self.hist_bins)
        ax.set_title(f'Actions hist, Nbins={self.hist_bins}, [{hist_[0]}, {hist_[-1]}]')
        self.logger.record("eval/figures/uhist", Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv"))
        plt.close()
    # =================================================================================

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            episode_rewards, true_lfp, actions_list = evaluate_policy_(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            
            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)

                self.eval_actions.append(actions_list)
                self.eval_lfp.append(true_lfp)

                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                results_dict = {'timesteps': self.evaluations_timesteps,
                               'rewards': self.evaluations_results,
                               'lfp': self.eval_lfp,
                               'actions': self.eval_actions,}
                np.save(self.log_path, results_dict)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards, ddof=1)
            cumsum_reward = np.sum(episode_rewards)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")

            # Add to current Logger Reward
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/cumsum_reward", float(cumsum_reward))

            # ========================================================================
            # log beta band power and PSD, action hist and energy
            bbpow = self.log_psd_for_envs(true_lfp)
            self.logger.record(f"eval/bbpow_mean", np.mean(bbpow))
            self.logger.record(f"eval/bbpow_sd", np.std(bbpow, ddof=1))

            u_energy = np.sum(np.abs(actions_list), axis=0)
            self.logger.record("eval/u_energy_mean", np.mean(u_energy))
            self.logger.record("eval/u_energy_sd", np.std(u_energy, ddof=1))
            self.log_hist_for_envs(actions_list)  # log actions hist
            # ========================================================================

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


################## For evaluation of non-RL algorithms

def evaluate_policy_hf_dbs(
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

    print('n_eval_episodes: ', n_eval_episodes, 'episode_count_targets: ', episode_count_targets)
    print(episode_counts, 'episode_counts')
    
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


def log_psd_for_simple_eval(sig_envs, psd_dt, cut_idx=1500, log_scale=True,
                     beta_a=12.5, beta_b=21):
    # for plotting
    reference_lfp = np.load('data/reference_fileterd_lfp.npy')
    sig_filt_R, _ = band_pass_envelope(reference_lfp, 1/psd_dt, order=2)
    ft_R = np.abs(np.fft.rfft(sig_filt_R)/sig_filt_R.shape[0])**2 * 2
    freq_R = np.fft.rfftfreq(sig_filt_R.shape[0], psd_dt)
    ft_R = filtfilt([1,1,1,1,1,1,1,1,1,1,1,1,], 5, ft_R)

    bbpow_list = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for ii, sig in enumerate(sig_envs):
        sig_filt, _ = band_pass_envelope(sig, 1/psd_dt, order=2)
        ft = np.abs(np.fft.rfft(sig_filt)/sig_filt.shape[0])**2 * 2
        freq = np.fft.rfftfreq(sig_filt.shape[0], psd_dt)
        ft = filtfilt([1,1,1,1,1,1,1,1,1,1,1,1,], 5, ft)
        if log_scale:
            ax.semilogy(freq[:cut_idx], ft[:cut_idx], label=f'with DBS, {ii}')
            # ax.set_ylim([10e-8, 10-2])
        else:
            ax.plot(freq[:cut_idx], ft[:cut_idx])
        # Also calculate beta band power
        idx = np.where((freq > beta_a) & (freq < beta_b))
        bbpow_list.append(np.sum(ft[idx]))
        
    ax.semilogy(freq_R[:cut_idx], ft_R[:cut_idx], label='noDBS')
    ax.set_xlabel('Hz')
    ax.set_ylabel('Volt**2/hz')
    # beta ranges: low (13–20 Hz) and high (21–35 Hz)
    ax.axvspan(4, 12.5, alpha=0.15, color='green', label='LF')
    ax.axvspan(12.5, 21, alpha=0.15, color='blue', label='Low beta band')
    ax.axvspan(21, 33.5, alpha=0.15, color='red', label='High beta band')
    ax.legend()
    ax.set_title(f'Power Spectral Density, beta-filt, log_scale')

    plt.savefig("data/tmp_images/psd.png")
    plt.close()
    return np.asarray(bbpow_list)


def evaluate_hf_dbs(model, eval_env: list, n_eval_episodes,
                    render=False, deterministic=True,
                    warn=False, callbacks_=None):
    reward_list = []
    bbpows_list, u_energy_list = [], []

    for env in eval_env:
        episode_rewards, true_lfp, actions_list = evaluate_policy_hf_dbs(
                    model, env,
                    n_envs=1,
                    n_eval_episodes=n_eval_episodes,
                    render=render,
                    deterministic=deterministic,
                    return_episode_rewards=True,
                    warn=warn, callback=callbacks_,)

        reward_list.append(episode_rewards[0])

        bbpow = log_psd_for_simple_eval(true_lfp, psd_dt=0.0005)  # NOTE: hardcode dt
        bbpows_list.append(bbpow[0])

        u_energy = np.sum(np.abs(actions_list), axis=0)
        u_energy_list.append(u_energy)
        
    bbpows_list = np.asarray(bbpows_list)
    u_energy_list = np.asarray(u_energy_list)
    reward_list = np.asarray(reward_list)

    mean_reward, std_reward = np.mean(reward_list), np.std(reward_list, ddof=1)

    print(f'Reward mean={mean_reward}, std={std_reward}')
    print(f'BBpow mean={np.mean(bbpows_list)}, std={np.std(bbpows_list, ddof=1)}')
    print(f'Energy cumsum={np.mean(u_energy_list)}, std={np.std(u_energy_list, ddof=1)}')
