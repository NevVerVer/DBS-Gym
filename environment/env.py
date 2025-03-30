import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Box 
import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

from environment.utils import (
    band_pass_envelope, calc_beta_band_power,
    remove_negative_w0, apply_locus_mask,
    create_distance_matrix, wavelet_kernel_matrix,
    units2sec, sec2units, create_directed_stim_masks)
# List is the same for all 3 version of env
from environment.env_configs.env1 import stim_rec_locus_coordinates  


def generate_perturbations(
    initial_vector: np.ndarray, 
    M: int = 10,
    step_scale: float = 0.1, # 10%
    random_seed: int = None
    ) -> np.ndarray:
    """
    Generate M stochastic perturbations of an initial vector, ensuring
    that the final vector does not drift too far from the initial vector.
    
    Args:
        initial_vector (np.ndarray): The original vector of numbers.
        M (int): Number of perturbations (steps) to generate.
        step_scale (float): Standard deviation factor for the random step.
        random_seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: An array of shape (M+1, n) containing the initial vector
                    followed by M perturbed vectors.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Store all perturbations; start with the initial vector
    perturbed_vectors = [initial_vector.copy()]
    vec_scale = np.std(initial_vector.copy(), ddof=1)
    
    for _ in range(M):
        current_vector = perturbed_vectors[-1]
        
        # Propose a small random step
        step = step_scale * vec_scale * np.random.randn(len(current_vector))
        new_vector = current_vector + step
        
        perturbed_vectors.append(new_vector)
    
    return np.array(perturbed_vectors)



class SimpleDBS:
    def __init__(self, 
                 grid_size,
                 distance_matrix,
                 elec_coords: list[list[int]],
                 rec_coords: list[list[int]],
                 neur_grid,
                 amplitudes: list[float] = 1.0,
                 directed_stimulation=False,
                 prc_type = 'I',
                 prc_scaling=1.0,
                 prc_mu=np.pi, 
                 prc_sigma=1.0,
                 naive=False,
                 logger=None):
        """
        Class for Deep Brain Stimulation electrode. The shape of the wave is set in Gym class.
        Can set multiple stimulation and recording contacts. Has directional stimulation
        option - for that the gym class observation can be changed to also adjust direction of stimulation. 

        :start_time: Time at which the first pulse should begin.
        :amplitude: Amplitude of the pulse, default is 1.
        :naive: wether or not scale dbs amplitude w.r.t. distance to neuron 
        """
        self.amplitudes = amplitudes
        self.prc_scaling = prc_scaling
        self.elec_idxs = []
        self.rec_idxs = []
        self.neur_grid = neur_grid

        assert len(self.amplitudes) == len(elec_coords), "Number of amplitudes is not equal to number of electrode coordinates!"

        for elec_coord in elec_coords:
            elec_idx = elec_coord[0] * grid_size[2]**2 + elec_coord[1] * grid_size[1] + elec_coord[2]
            self.elec_idxs.append(elec_idx)
        for rec_coord in rec_coords:
            rec_idx = rec_coord[0] * grid_size[2]**2 + rec_coord[1] * grid_size[1] + rec_coord[2]
            self.rec_idxs.append(rec_idx)        
        
        if logger:
            for elec_idx, elec_coord in zip(self.elec_idxs, elec_coords):
                logger.info(f'Electrode idx: {elec_idx}, Elec coordinate: {elec_coord}')
            for rec_idx, rec_coord in zip(self.rec_idxs, rec_coords):
                logger.info(f'Electrode idx: {rec_idx}, Elec coordinate: {rec_coord}')
        
        # Calculate conductances for stimulating electrodes
        self.dist_vectors = []
        self.conductances = []
        for elec_idx in self.elec_idxs:
            dist_vector = distance_matrix[elec_idx]
            conductance = 1 - dist_vector 
            conductance = np.where(conductance < 0.0, 0, conductance)
            if naive:
                if logger:
                    logger.info('Naive version of DBS!')
                else:
                    print('Naive version of DBS!')
                conductance = np.ones_like(dist_vector)
            self.dist_vectors.append(dist_vector)
            self.conductances.append(conductance)
            if not directed_stimulation:
                print(f'DBS affects {np.argwhere(conductance > 0.0).shape[0]} neurons, min={round(conductance.min(), 3)} & max={round(conductance.max(), 3)}')

        # Calculate directional kernels
        self.directional_masks_list = []
        if directed_stimulation:
            print('CAUTIOUS! DIRECTIONAL STIMULATION IS TURNED ON')
            for elec_coord in elec_coords:
                mask1, mask2, mask3 = create_directed_stim_masks(self.neur_grid, 
                                                                np.asarray(elec_coord), elec_idx)
                self.directional_masks_list.append([mask1, mask2, mask3])
            # By default choose mask with 1st direction
            self.directional_mask = [d[0] for d in self.directional_masks_list]  
        
            self.conductances_ = []
            for c, d in zip(self.conductances, self.directional_mask):
                new_cond = c*d
                self.conductances_.append(new_cond)
                print(f'DIRECTIONAL DBS affects {np.argwhere(new_cond > 0.0).shape[0]} neurons, min={conductance.min()} & max={conductance.max()}')
            self.conductances = self.conductances_

        # Calculate conductances for recording electrodes
        self.rec_dist_vectors = []
        self.rec_conductances = []
        for rec_idx in self.rec_idxs:
            dist_vector = distance_matrix[rec_idx]
            conductance = 1 - dist_vector
            conductance = np.where(conductance < 0.0, 0, conductance)
            if naive:
                if logger:
                    logger.info('Naive version of DBS!')
                else:
                    print('Naive version of DBS!')
                conductance = np.ones_like(dist_vector)
            self.rec_dist_vectors.append(dist_vector)
            self.rec_conductances.append(conductance)

        # Neuron response properties
        self.prc_mu = prc_mu
        self.prc_sigma = prc_sigma
        self.prc_type = prc_type
        if prc_type == 'I':
            self.prc = self.prc_type_I
        elif prc_type == 'II':
            self.prc = self.prc_type_II
        elif prc_type == 'Gaussian':
            self.prc = self.prc_gaussian
        elif prc_type == 'dummy':
            self.prc = self.prc_dummy
        else: 
            raise ValueError('Wrong type of PRC function!')

    def prc_dummy(self, theta):
        return self.prc_scaling * np.ones_like(theta)

    def prc_type_I(self, theta):
        return self.prc_scaling * (1 - np.cos(theta))
    
    def prc_type_II(self, theta):
        return self.prc_scaling * np.sin(theta)
    
    def prc_gaussian(self, theta):
        return self.prc_scaling * np.exp(-((theta - self.prc_mu) ** 2) / (2 * self.prc_sigma ** 2))


class KuramotoJAX:
    """
    Class of the Kuramoto model implemented in jax. 
    Only positive values are okay for w0
    """
    def __init__(self, 
                 n_neurons, 
                 K: Union[float, int, np.ndarray], 
                 grid_size,
                 w0,
                 neur_coords, 
                 neur_grid,
                 electrode_coords, 
                 recorders_coords,
                 conduct_modifier, 
                 spatial_kernel='cos',
                 wavelet_amp = 1.0,
                 wavelet_steepness = 1.0,
                 directed_stimulation=False,
                 electrode_amps: list[float] = [1.0, 1.0],
                 electrode_prc_type='I',
                 electrode_prc_scaling=0.5,
                 naive_dbs=False,
                 logger=None,
                 ):
        self.K = K
        self.n_neurons = n_neurons
        self.w0 = remove_negative_w0(w0)
        assert np.min(self.w0) >= 0, "Natural frequencies w0 must be positive!"

        self.grid_size = grid_size
        self.neur_coords, self.neur_grid = neur_coords, neur_grid

        self.distance_matrix = create_distance_matrix(self.neur_coords)

        self.spatial_kernel = spatial_kernel
        if self.spatial_kernel == 'cos':
            self.alpha = np.cos(self.distance_matrix)
        elif self.spatial_kernel == 'wavelet':
            self.alpha = wavelet_kernel_matrix(self.distance_matrix,
                                       amplitude=wavelet_amp,
                                       steepness=wavelet_steepness)
        else:
            raise ValueError(f'Wrong distance matrix type: {self.distance_matrix}')
           
        # Initialize electrode
        electrode_distances = self.neur_grid * conduct_modifier
        self.dbs = SimpleDBS(grid_size,
                             distance_matrix=create_distance_matrix(electrode_distances),
                             elec_coords=electrode_coords, # literally x, y (as in cartesian plot)
                             rec_coords=recorders_coords,
                             neur_grid=self.neur_grid,
                             directed_stimulation=directed_stimulation,
                             amplitudes=electrode_amps,
                             prc_type=electrode_prc_type,
                             prc_scaling=electrode_prc_scaling,
                             naive=naive_dbs,
                             logger=logger)
        self.pulse = np.zeros((self.n_neurons))

        # Define all needed for differential eq. solver
        self.ode_term = ODETerm(self.dynamics)
        self.solver = Dopri5()
        self.stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)


    def dynamics(self, t, y, args):
        theta = jnp.fmod(y, 2*jnp.pi)
        dtheta = args[0] + args[1] *\
            jnp.sum(args[3] * jnp.sin(theta - jnp.tile(theta, (args[2], 1)).T), axis=1) + args[4]
        return dtheta


    # @jax.jit
    def forward(self, t_eval, state0):
        solution = diffeqsolve(
                            self.ode_term,
                            self.solver,
                            args=(self.w0, self.K/self.n_neurons, self.n_neurons,
                                  jnp.array(self.alpha), self.pulse),
                            t0=t_eval[0], t1=t_eval[-1],
                            dt0=0.05,
                            y0=state0,
                            saveat=SaveAt(ts=t_eval),
                            stepsize_controller=self.stepsize_controller)
        return solution.ys


class SpatialKuramoto(gym.Env):
  metadata = {'render.modes': ['human']}
    
  def __init__(self, params_dict, save_init=False):
    """ 
    Class for environment for RL aDBS training.

    len: in [units] - pseudo-seconds
    sec: in [seconds] - need for stats
    counts: in [counts]
    idxs: in [idx], shape
    """
    super().__init__()
    self.save_init = save_init
    self.params_dict = params_dict
    self.reset_count = -1
    self.verbose = params_dict['verbose']
    np.random.seed(self.params_dict['rand_seed'])

    # period == step
    self.step_len = self.params_dict['electrode_width'] + self.params_dict['electrode_pause']  # units + units = units

    self.observe_wind_len = self.step_len * self.params_dict['observe_wind_counts']  # units * counts = units
    self.observe_wind_idxs = int(self.observe_wind_len / self.params_dict['verbose_dt'])   # [counts]

    self.total_episode_len = self.params_dict['total_episode_len']    # [units]
    self.total_episode_counts = int(self.total_episode_len / self.step_len)  # counts

    self.transient_state_len = self.params_dict['transient_state_len']  # [units]
    if self.transient_state_len < self.observe_wind_len:
      raise ValueError('Transient state should be longer than RL agent observation window!')

    ### For RL Agent  
    self.dim = 1  # dimensionality of our observation space
    self.dbs_action_bounds = self.params_dict['dbs_action_bounds']
    self.ppo_action_bounds = [-1., 1.]
    self.action_space = Box(low=self.ppo_action_bounds[0],
                            high=self.ppo_action_bounds[1],
                            shape=(1,), dtype=np.float32)  # for square wave 
    self.observation_space = Box(low=-1.5, high=1.5,
                                 shape=(1, self.observe_wind_idxs,),
                                 dtype=np.float32)
    self.done = False
    self.current_step = 0
    self.current_time = 0.
    self.theta_state = np.empty((1, self.observe_wind_idxs))  # state for RL agent, [1, self.observe_wind_idxs]
    self.sol_state = []   # use as init state for steps, for verbose, shape: [self.observe_wind_idxs, n_neurons]

    # Choose reward function
    if self.params_dict['reward_func'] == 'bbpow_action':
        self.reward_func = self.reward_bbpow_action
    elif self.params_dict['reward_func'] == 'temp_const_action':
        self.reward_func = self.reward_temp_const_lfp_betafilt_action
    elif self.params_dict['reward_func'] == 'bbpow_threth_action':
        self.reward_func = self.reward_bbpow_threth_action
    else: 
        raise ValueError('Wrong reward function!')
    
    # Choose recording function
    if self.params_dict['recording_kernel'] == 'naive':
        self.calc_lfp = self.calc_naive_lfp
    elif self.params_dict['recording_kernel'] == 'gaussian':
        self.calc_lfp = self.calc_distance_lfp
    else: 
        raise ValueError('Wrong recording kernel function!')
    
    self.K = params_dict['K']
    self.w0 = params_dict['w0']
    self.w0_without_locus = params_dict['w0_without_locus']
    self.w0_without_locus_ = deepcopy(params_dict['w0_without_locus'])

    self.elec_coords = params_dict['elec_coords']
    self.rec_coords = params_dict['rec_coords']
    
    self.save_events = params_dict['save_events']
    self.encapsulation_coeff = params_dict['conduct_modifier']

    ### Temporal drift parameters (For env2)
    if params_dict['temporal_drift']:
        self.random_freq_update = params_dict['random_freq_update']  # Turn on for training

        if self.save_events:  # For logging times of events for evaluation
            self.temporal_events = {'electrode_drift': [],
                                    'encapsulation_drift': [],
                                    'plasticity_drift': [],
                                    'mov_modulation_drift': []}  
        # Electrode drift parameters
        self.elec_drift_episode = params_dict['electrode_drift_freq']
        self.elec_encaps_episode = params_dict['encapsulation_drift_freq']
        self.encaps_precent = params_dict['encapsulation_percent']
        self.mov_mod_episode = params_dict['mov_modulation_drift_freq']

        # Plasticity drfit parameters
        self.plasticity_episode = params_dict['plasticity_drift_freq']
        assert self.plasticity_episode >=2, "Maybe set plasticity drift more rarely?"
        self.plasticity_percent = params_dict['plasticity_percent']
        self.reset_plasticity_episode = params_dict['reset_plasticity_episode']
        self.plasticity_process_count = 0

        # Generate plasticity drift process (w0)
        self.rng = np.random.default_rng(seed=params_dict['rand_seed'])
        self.w0_process = generate_perturbations(self.w0_without_locus,
                                                 M=self.reset_plasticity_episode * 2,
                                                 step_scale=self.plasticity_percent*0.01,)
    else:
        print('No temporal drift events!')

    ### Spatial parameters
    self.spatial_events = []
    self.spatial_var_freq = params_dict['spatial_var_freq']
    self.spatial_var_episode = self.spatial_var_freq

    self.reset()


  def rescale_action(self, action):
    x, y = self.ppo_action_bounds  # original_range
    z, k = self.dbs_action_bounds  # target_range
    new_action = z + ((k - z) * (action - x)) / (y - x)
    return new_action


  def calc_naive_lfp(self, sig):
    """ 
    Calculate true LFP of the population.
    sig: tensor of size [time x neurons] of phases in rad
    """
    return np.mean(np.cos(sig), axis=1)


  def calc_distance_lfp(self, sig) -> list:
    """
    sig: tensor of size [time x neurons] of phases in rad
    returns: vector of distance-accounted lfp-s from all recording electrodes
    """
    records = np.zeros((sig.shape[0]))
    for conductance in self.kuramoto.dbs.rec_conductances:
        records += np.mean(np.cos(sig) * conductance, axis=1)
    return records


  def step(self, action):
    """
    Simple square Wave DBS. 
    """
    self.u = [self.rescale_action(float(a)) for a in action] # amplitudes
    ### 1. DBS ON. I part
    pulse = np.zeros((self.params_dict['num_oscillators']))
    for amplitude, conductance in zip(self.u, self.kuramoto.dbs.conductances):
      pulse += conductance * amplitude
    self.kuramoto.pulse = pulse

    self.t_eval_step_I = np.arange(self.current_time,
                                   self.current_time + self.params_dict['electrode_width'],
                                   self.params_dict['verbose_dt'])   # == verbose dt
    self.sol_state = self.kuramoto.forward(self.t_eval_step_I, self.sol_state[-1, :])
    self.sol_state_ = self.sol_state
    self.current_time = self.t_eval_step_I[-1]

    ### 2. DBS OFF. II part
    self.kuramoto.pulse = np.zeros((self.params_dict['num_oscillators']))
    self.t_eval_step_II = np.arange(self.current_time,
                                    self.current_time + self.params_dict['electrode_pause'],
                                    self.params_dict['verbose_dt'])   # == verbose dt
    
    self.sol_state = self.kuramoto.forward(self.t_eval_step_II, self.sol_state[-1, :])
    self.sol_state_ = np.concatenate([self.sol_state_ , self.sol_state])
    self.current_time = self.t_eval_step_II[-1]

    # Calcualte observation 
    self.theta_mean = self.calc_naive_lfp(self.sol_state_[:-1, :])  # True LFP
    self.theta_records = self.calc_lfp(self.sol_state_[:-1, :])   # As we set in init

    self.theta_state = np.append(self.theta_state, self.theta_records[np.newaxis,...], axis=1)
    self.theta_state = self.theta_state[:, -self.observe_wind_idxs:]

    self.current_step += 1
    self.done = self.current_step >= self.total_episode_counts
    self.reward_ = self.reward_func(self.theta_state[0], self.u)
    
    return (self.theta_state.astype(np.float32), self.reward_, self.done, False, {})


  def calc_next_event(self, f, deltas=[-1, 0, 1]):
    if self.random_freq_update:
        # Update new drift event time. Random event
        range_list = [f+f_delta for f_delta in deltas]
        new_event = np.random.choice(range_list)
        return new_event
    else:
        return f


  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):  
    """
    We update temporal and spatial features parameters during reset. 
    """
    super().reset(seed=seed)

    self.current_step = 0
    self.current_time = 0.
    self.done = False
    self.reset_count += 1  # +1 to reset counter
    self.states = []
    self.actions = []
    params_dict = self.params_dict
    self.theta_state = np.empty((1, self.observe_wind_idxs), dtype=np.float32)

    ### Temporal Features Update
    if params_dict['temporal_drift']:

        if self.elec_drift_episode == self.reset_count:
            self.elec_drift_episode += self.calc_next_event(params_dict['electrode_drift_freq'],
                                                            [-1, 0, 1])
            new_coords_ = [[10000, 0, 0]]
            bound1, bound2 = 1, min(self.params_dict['grid_size'])-2  # Remove bound coords
            # Check out-of-bounds movement. TODO: Now it works for now only 1 contact!
            while any([coord < bound1 or coord > bound2 for coord in new_coords_[0]]):
                # Sample delta and update electrode coordinates
                elec_delta = np.empty(3)
                for i in range(3):
                    elec_delta[i] = np.random.choice([-1, 1]) * np.random.choice([0, 1])
                new_coords_ = np.asarray(self.elec_coords + elec_delta)
                new_coords_ = new_coords_.astype(int).tolist()
            self.elec_coords = new_coords_
            
            # Log and print
            if self.save_events:
                self.temporal_events['electrode_drift'].append([self.reset_count, self.elec_coords])
            if self.verbose:
                print(f'Electode drift! Changed electrode location to {self.elec_coords}')

        if self.elec_encaps_episode == self.reset_count:
            self.elec_encaps_episode += self.calc_next_event(params_dict['encapsulation_drift_freq'],
                                                             [-2, -1, 0, 1, 2])
            self.encapsulation_coeff += self.encaps_precent  # We accumulate encapsulation with each step

            # Log and print
            if self.save_events:
                self.temporal_events['encapsulation_drift'].append([self.reset_count, self.encaps_precent])
            if self.verbose:
                print(f'Electode encapsulation! Reduced electrode conductances by {self.encapsulation_coeff}')
                cond_ = self.kuramoto.dbs.conductances[0]
                print(f'NOW DBS affects {np.argwhere(cond_ > 0.0).shape[0]} neurons, min={round(cond_.min(), 3)} & max={round(cond_.max(), 3)}')

        if self.plasticity_episode == self.reset_count:
            self.plasticity_episode += self.calc_next_temp_event(params_dict['plasticity_drift_freq'],
                                                                  [0, 1])
            self.w0_without_locus = self.w0_process[self.plasticity_process_count]
            self.plasticity_process_count += 1

            # Log and print
            if self.save_events:
                self.temporal_events['plasticity_drift'].append([self.reset_count, self.w0_without_locus])
            if self.verbose:
                print(f'Drift of w0 by {self.plasticity_percent}%, to {self.plasticity_process_count}')

        # Reset plasticity and generate stoch. process anew
        if self.reset_count % self.reset_plasticity_episode == 0:
            if self.verbose:
                print(f'Reseting plastisity...')
  
            self.plasticity_process_count = 0
            self.w0_without_locus = deepcopy(self.w0_without_locus_)
            # Generate w0 drift process
            self.w0_process = generate_perturbations(self.w0_without_locus,
                                                     M=self.reset_plasticity_episode * 2,
                                                     step_scale=self.plasticity_percent*0.01,)
                
    ### Spatial Features Update.
    if params_dict['spatial_feature']:
        if self.spatial_var_episode == self.reset_count and self.reset_count > 2:
            # Vary location of coordinates of electrode
            index = np.random.choice(len(stim_rec_locus_coordinates))
            self.elec_coords = [stim_rec_locus_coordinates[index][0]]
            self.rec_coords = [stim_rec_locus_coordinates[index][1]]
            # self.locus_coords = stim_rec_locus_coordinates[index][2]  # TODO: fix
  
            self.spatial_var_episode += self.spatial_var_freq

            # Log and print
            self.spatial_events.append([self.reset_count, stim_rec_locus_coordinates[index]])
            if self.verbose:
                print('Reinit spatial parameters! New coordinates are: ', stim_rec_locus_coordinates[index])

    # Save all temporal and spatial features statistics
    if self.params_dict['save_events'] and self.params_dict['log_path'] is not None and self.reset_count > 1:
        log_name = os.path.join(self.params_dict['log_path'], f"temp_{self.reset_count}.npy")
        np.save(log_name, self.temporal_events)


    ### Re-calculate w0 with or without plasticity 
    self.w0 = apply_locus_mask(self.w0_without_locus,
                               params_dict['locus_without_w0'],
                               params_dict['locus_mask'],)
    ### Re-init model
    self.kuramoto = KuramotoJAX(
            n_neurons=            params_dict['num_oscillators'],
            K=                    params_dict['K'],
            grid_size=            params_dict['grid_size'],

            w0 =                  self.w0,
            neur_coords=          params_dict['neur_coords'],
            neur_grid=            params_dict['neur_grid'],

            spatial_kernel=       params_dict['spatial_kernel'],
            wavelet_amp=          params_dict['wavelet_amp'],
            wavelet_steepness=    params_dict['wavelet_steepness'],

            # DBS params
            directed_stimulation= params_dict['directed_stimulation'],  # bool
            electrode_coords=     self.elec_coords,
            recorders_coords=     self.rec_coords,
            conduct_modifier=     self.encapsulation_coeff,  # the bigger, the less is DBS kernel

            electrode_amps=       params_dict['electrode_amps'],
            electrode_prc_scaling=params_dict['electrode_prc_scaling'],
            electrode_prc_type=   params_dict['electrode_prc_type'],
            naive_dbs =           params_dict['naive_dbs'],
    )
    if not self.save_init:
      self.init_state = np.random.normal(loc=params_dict['init_state_mean'],
                                        scale=params_dict['init_state_sd'],
                                        size=(params_dict['num_oscillators']))
      self.init_state = remove_negative_w0(self.init_state)

    # for logging
    self.kw0 = self.kuramoto.w0     
    self.kneur_grid = self.kuramoto.neur_grid
    self.kgrid_size = self.kuramoto.grid_size

    ### Run transient state
    self.t_eval_transient = np.arange(self.current_time,
                                      self.transient_state_len,
                                      params_dict['verbose_dt'])   # [units]
    self.current_time = self.t_eval_transient[-1]   # [units]
    self.sol_state = self.kuramoto.forward(self.t_eval_transient, self.init_state)
    self.theta_record_transient = self.calc_lfp(self.sol_state[:-1, :])
    self.theta_state = self.theta_record_transient[-self.observe_wind_idxs:][np.newaxis, ...]

    return self.theta_state.astype(np.float32), {}
  
  
  def render(self, mode='human', close=False):
    pass
  
  
  def close(self):
    pass
  
  
  def calculate_bbpow(self, solutions):
    sig = np.concatenate(solutions)
    beta_band = [12.5, 21]    # Low beta band
    psd_dt = units2sec(self.params_dict['verbose_dt'])
    bbpow = calc_beta_band_power(sig, psd_dt,
                                    beta_band[0], beta_band[1])
    return bbpow


  def calculate_energy(self, actions):
    return np.abs(actions).sum()
  

  def reward_bbpow_action(self, x_state, action_value, baseline=False):
    """
    #1 REWARD = - calc_beta_band_power (in window): float
    For this reward we need huge width of window or freq. leakege happens
    """
    assert len(x_state.shape) == 1, "Incorrect dimension of theta_state"
    beta_band = [12.5, 21]    # Low beta band
    alpha, beta = 1e4, 1e-2

    psd_dt = units2sec(self.params_dict['verbose_dt'])
    r1 = alpha * calc_beta_band_power(x_state, psd_dt,
                                      beta_band[0], beta_band[1])
    return - r1 - beta * np.abs(action_value[0])
  

  def reward_temp_const_lfp_betafilt_action(self, x_state,
                                 action_value, baseline=False):
    """
    #2 as in Krylov et al., 2021 paper
    """
    assert len(x_state.shape) == 1, "Incorrect dimension of theta_state"

    alpha, beta = 1e3, 1e-2

    psd_dt = units2sec(self.params_dict['verbose_dt'])
    x_state_filt, _ = band_pass_envelope(x_state, 1/psd_dt, order=2)
    r1 = alpha * (x_state_filt[-1] - np.mean(x_state_filt))**2
    r2 = beta * np.abs(action_value[0])
    return -r1 - r2
  

  def reward_bbpow_threth_action(self, x_state,
                                 action_value, baseline=False):
    """
    #3 As in Gao paper. If bbpow higher than A, then reward=0.
    Else reward=1.
    """
    assert len(x_state.shape) == 1, "Incorrect dimension of theta_state"
    bbpow_coeff, bbpow_threshold = 5., 20
    beta_band = [12.5, 21]  # Low beta band
    alpha, beta = 1e4, 1e-2

    psd_dt = units2sec(self.params_dict['verbose_dt'])
    bbpow = alpha * calc_beta_band_power(x_state, psd_dt,
                                        beta_band[0], beta_band[1])
    if bbpow > bbpow_threshold:
        r1 = bbpow_coeff
    else:
        r1 = 0
    u = np.abs(float(action_value[0]))
    return - r1 - u
