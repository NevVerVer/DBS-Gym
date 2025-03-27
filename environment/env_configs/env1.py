import numpy as np

# Spatial variation 
stim_rec_locus_coordinates = [
        [[5, 2, 3], [3, 5, 1], [1, 2, 3]],
        [[4, 3, 1], [2, 5, 4], [2, 1, 4]],
        [[4, 3, 6], [2, 6, 4], [4, 3, 2]],
        [[5, 2, 1], [3, 5, 3], [5, 2, 5]],
        [[1, 3, 2], [4, 1, 4], [4, 5, 4]],
        [[6, 6, 4], [4, 4, 3], [3, 6, 5]],
        [[6, 5, 3], [1, 6, 4], [3, 2, 6]],
        [[6, 3, 5], [4, 1, 1], [5, 6, 1]],
        [[6, 5, 4], [1, 6, 3], [3, 2, 1]],
        [[4, 5, 3], [3, 3, 1], [6, 4, 1]],
        [[2, 3, 2], [4, 5, 3], [1, 5, 4]],
        [[5, 3, 2], [5, 5, 4], [5, 2, 5]],
        [[1, 6, 2], [6, 5, 1], [3, 2, 4]],
        [[2, 3, 3], [3, 3, 6], [1, 1, 5]],
        [[3, 5, 2], [1, 6, 4], [1, 3, 3]],
        ]


n_neurons = 512
grid_size = [8, 8, 8]
coord_modif = 0.1
locus_center = [4, 4, 4]
locus_size = 0.55

params_dict_train = {
            'logger_name': 'k',
            'log_path': None,
            'rand_seed': 10,
            'verbose': 1,
        
            # Model parameters
            'model_type': '2dspatial',
            'K': 0.52,
            'num_oscillators': n_neurons,
            'grid_size': grid_size,

            'w0': None, # rad/sec
            'wmuL': 17,
            'wsdL': 1,
            'neur_coords': None,
            'neur_grid': None,
            'coord_modif': coord_modif,
            'spatial_kernel': 'cos',  
            'wavelet_amp': 1.0,
            'wavelet_steepness': 0.6,
        
            # DBS (RL agent) parameters
            'elec_coords':[[4, 3, 4]],
            'rec_coords':[[1, 1, 1]],
            'directed_stimulation': False,
            'conduct_modifier': 0.1,  # the bigger, the less is kernel
            'recording_kernel': 'gaussian',
        #     'reinit_spatial_params': True,
            'locus_size': locus_size,
            'locus_center': locus_center,
            
            # (0.012, 0.988) - irl params 
            'transient_state_len': 200.,  # [units]
            'electrode_width': 0.15,    # [units]
            'electrode_pause': 0.75,    # [units] 
            'electrode_amps':  [0.],   ## this is RL action   # V
            'dbs_action_bounds': [-5, 5], 

            'electrode_prc_scaling': 1.0,    #
            'electrode_prc_type': 'dummy',
            'naive_dbs': False,
        
            # Stimulation parameters
            'verbose_dt': 0.05, # 0.05  [units]
            'total_episode_len': 5000,  # [units],
            'reward_func': None,
            'observe_wind_counts': 130,  # [counts], how many periods we show to RL agent in sliding window

            'init_state_type': 'normal',
            'init_state_mean': np.pi,
            'init_state_sd': 0.6,
            
            # Temporal features parameters
            'temporal_drift': False,
            'random_freq_update': True,
            'save_events': False,  # to save times of temporal drift
            'electrode_drift_freq': 0,
            'plasticity_drift_freq': 0,
            'plasticity_percent': 0,   # [%]
            'reset_plasticity_episode': 0,

            'encapsulation_drift_freq': 0,
            'encapsulation_percent': 0,    # [%]
            'mov_modulation_drift_freq': 0,

            # Spatial features parameters
            'spatial_feature': True,
            'spatial_var_freq': 10, # for train, how freq-ly we vary electrode coordinates  and RAND_SEED
    }

# ============================================================

eval0 = {
            'logger_name': 'k',
            'log_path': None,
            'rand_seed': 10,
            'verbose': 1,
        
            # Model parameters
            'model_type': '2dspatial',
            'K': 0.52,
            'num_oscillators': n_neurons,
            'grid_size': grid_size,

            'w0': None, # rad/sec
            'wmuL': 17,
            'wsdL': 1,
            'neur_coords': None,
            'neur_grid': None,
            'coord_modif': coord_modif,
            'spatial_kernel': 'cos',  
            'wavelet_amp': 1.0,
            'wavelet_steepness': 0.6,
        
            # DBS (RL agent) parameters
            'elec_coords': [stim_rec_locus_coordinates[0][0]],
            'rec_coords': [stim_rec_locus_coordinates[0][1]],
            'directed_stimulation': False,
            'conduct_modifier': 0.1,  # the bigger, the less is kernel
            'recording_kernel': 'gaussian',
        #     'reinit_spatial_params': False,
            'locus_size': locus_size,
            'locus_center': stim_rec_locus_coordinates[0][2],
            
            # (0.012, 0.988) - irl params 
            'transient_state_len': 200.,  # [units]
            'electrode_width': 0.15,    # [units]
            'electrode_pause': 0.75,    # [units] 
            'electrode_amps':  [0.],   ## this is RL action   # V
            'dbs_action_bounds': [-5, 5], 

            'electrode_prc_scaling': 1.0,    #
            'electrode_prc_type': 'dummy',
            'naive_dbs': False,
        
            # Stimulation parameters
            'verbose_dt': 0.05,  # [units]
            'total_episode_len': 1000,  # [units],
            'reward_func': None,
            'observe_wind_counts': 130,  # [counts], how many periods we show to RL agent in sliding window

            'init_state_type': 'normal',
            'init_state_mean': np.pi,
            'init_state_sd': 0.6,

            # Temporal features parameters
            'temporal_drift': False,
            'random_freq_update': True,
            'save_events': False,  # to save times of temporal drift
            'electrode_drift_freq': 0,
            'plasticity_drift_freq': 0,
            'plasticity_percent': 0,   # [%]
            'reset_plasticity_episode': 0,

            'encapsulation_drift_freq': 0,
            'encapsulation_percent': 0,    # [%]
            'mov_modulation_drift_freq': 0,

            # Spatial features parameters
            'spatial_feature': False,
            'spatial_var_freq': 0, # for train, how freq-ly we vary electrode coordinates  and RAND_SEED
    }

eval1 = {
            'logger_name': 'k',
            'log_path': None,
            'rand_seed': 10,
            'verbose': 1,
        
            # Model parameters
            'model_type': '2dspatial',
            'K': 0.52,
            'num_oscillators': n_neurons,
            'grid_size': grid_size,

            'w0': None, # rad/sec
            'wmuL': 17,
            'wsdL': 1,
            'neur_coords': None,
            'neur_grid': None,
            'coord_modif': coord_modif,
            'spatial_kernel': 'cos',  
            'wavelet_amp': 1.0,
            'wavelet_steepness': 0.6,
        
            # DBS (RL agent) parameters
            'elec_coords': [stim_rec_locus_coordinates[1][0]],
            'rec_coords': [stim_rec_locus_coordinates[1][1]],
            'directed_stimulation': False,
            'conduct_modifier': 0.1,  # the bigger, the less is kernel
            'recording_kernel': 'gaussian',
        #     'reinit_spatial_params': False,
            'locus_size': locus_size,
            'locus_center': stim_rec_locus_coordinates[1][2],

            # (0.012, 0.988) - irl params 
            'transient_state_len': 200.,  # [units]
            'electrode_width': 0.15,    # [units]
            'electrode_pause': 0.75,    # [units] 
            'electrode_amps':  [0.],   ## this is RL action   # V
            'dbs_action_bounds': [-5, 5], 

            'electrode_prc_scaling': 1.0,    #
            'electrode_prc_type': 'dummy',
            'naive_dbs': False,
        
            # Stimulation parameters
            'verbose_dt': 0.05,  # [units]
            'total_episode_len': 1000,  # [units],
            'reward_func': None,
            'observe_wind_counts': 130,  # [counts], how many periods we show to RL agent in sliding window

            'init_state_type': 'normal',
            'init_state_mean': np.pi,
            'init_state_sd': 0.6,

            # Temporal features parameters
            'temporal_drift': False,
            'random_freq_update': True,
            'save_events': False,  # to save times of temporal drift
            'electrode_drift_freq': 0,
            'plasticity_drift_freq': 0,
            'plasticity_percent': 0,   # [%]
            'reset_plasticity_episode': 0,

            'encapsulation_drift_freq': 0,
            'encapsulation_percent': 0,    # [%]
            'mov_modulation_drift_freq': 0,

            # Spatial features parameters
            'spatial_feature': False,
            'spatial_var_freq': 0, # for train, how freq-ly we vary electrode coordinates  and RAND_SEED
    }

eval2 = {
            'logger_name': 'k',
            'log_path': None,
            'rand_seed': 10,
            'verbose': 1,
        
            # Model parameters
            'model_type': '2dspatial',
            'K': 0.52,
            'num_oscillators': n_neurons,
            'grid_size': grid_size,

            'w0': None, # rad/sec
            'wmuL': 17,
            'wsdL': 1,
            'neur_coords': None,
            'neur_grid': None,
            'coord_modif': coord_modif,
            'spatial_kernel': 'cos',  
            'wavelet_amp': 1.0,
            'wavelet_steepness': 0.6,
        
            # DBS (RL agent) parameters
            'elec_coords': [stim_rec_locus_coordinates[2][0]],
            'rec_coords': [stim_rec_locus_coordinates[2][1]],
            'directed_stimulation': False,
            'conduct_modifier': 0.1,  # the bigger, the less is kernel
            'recording_kernel': 'gaussian',
        #     'reinit_spatial_params': False,
            'locus_size': locus_size,
            'locus_center': stim_rec_locus_coordinates[2][2],
            
            # (0.012, 0.988) - irl params 
            'transient_state_len': 200.,  # [units]
            'electrode_width': 0.15,    # [units]
            'electrode_pause': 0.75,    # [units] 
            'electrode_amps':  [0.],   ## this is RL action   # V
            'dbs_action_bounds': [-5, 5], 

            'electrode_prc_scaling': 1.0,    #
            'electrode_prc_type': 'dummy',
            'naive_dbs': False,
        
            # Stimulation parameters
            'verbose_dt': 0.05, # 0.05  [units]
            'total_episode_len': 1000,  # [units],
            'reward_func': None,
            'observe_wind_counts': 130,  # [counts], how many periods we show to RL agent in sliding window

            'init_state_type': 'normal',
            'init_state_mean': np.pi,
            'init_state_sd': 0.6,

            # Temporal features parameters
            'temporal_drift': False,
            'random_freq_update': True,
            'save_events': False,  # to save times of temporal drift
            'electrode_drift_freq': 0,
            'plasticity_drift_freq': 0,
            'plasticity_percent': 0,   # [%]
            'reset_plasticity_episode': 0,

            'encapsulation_drift_freq': 0,
            'encapsulation_percent': 0,    # [%]
            'mov_modulation_drift_freq': 0,

            # Spatial features parameters
            'spatial_feature': False,
            'spatial_var_freq': 0, # for train, how freq-ly we vary electrode coordinates  and RAND_SEED
    }

eval3 = {
            'logger_name': 'k',
            'log_path': None,
            'rand_seed': 10,
            'verbose': 1,
        
            # Model parameters
            'model_type': '2dspatial',
            'K': 0.52,
            'num_oscillators': n_neurons,
            'grid_size': grid_size,

            'w0': None, # rad/sec
            'wmuL': 17,
            'wsdL': 1,
            'neur_coords': None,
            'neur_grid': None,
            'coord_modif': coord_modif,
            'spatial_kernel': 'cos',  
            'wavelet_amp': 1.0,
            'wavelet_steepness': 0.6,
        
            # DBS (RL agent) parameters
            'elec_coords': [stim_rec_locus_coordinates[3][0]],
            'rec_coords': [stim_rec_locus_coordinates[3][1]],
            'directed_stimulation': False,
            'conduct_modifier': 0.1,  # the bigger, the less is kernel
            'recording_kernel': 'gaussian',
        #     'reinit_spatial_params': False,
            'locus_size': locus_size,
            'locus_center': stim_rec_locus_coordinates[3][2],
            
            # (0.012, 0.988) - irl params 
            'transient_state_len': 200.,  # [units]
            'electrode_width': 0.15,    # [units]
            'electrode_pause': 0.75,    # [units] 
            'electrode_amps':  [0.],   ## this is RL action   # V
            'dbs_action_bounds': [-5, 5],

            'electrode_prc_scaling': 1.0,    #
            'electrode_prc_type': 'dummy',
            'naive_dbs': False,
        
            # Stimulation parameters
            'verbose_dt': 0.05,  # [units]
            'total_episode_len': 1000,  # [units],
            'reward_func': None,
            'observe_wind_counts': 130,  # [counts], how many periods we show to RL agent in sliding window

            'init_state_type': 'normal',
            'init_state_mean': np.pi,
            'init_state_sd': 0.6,

            # Temporal features parameters
            'temporal_drift': False,
            'random_freq_update': True,
            'save_events': False,  # to save times of temporal drift
            'electrode_drift_freq': 0,
            'plasticity_drift_freq': 0,
            'plasticity_percent': 0,   # [%]
            'reset_plasticity_episode': 0,

            'encapsulation_drift_freq': 0,
            'encapsulation_percent': 0,    # [%]
            'mov_modulation_drift_freq': 0,

            # Spatial features parameters
            'spatial_feature': False,
            'spatial_var_freq': 0, # for train, how freq-ly we vary electrode coordinates  and RAND_SEED
    }

eval4 = {
            'logger_name': 'k',
            'log_path': None,
            'rand_seed': 10,
            'verbose': 1,
        
            # Model parameters
            'model_type': '2dspatial',
            'K': 0.52,
            'num_oscillators': n_neurons,
            'grid_size': grid_size,

            'w0': None, # rad/sec
            'wmuL': 17,
            'wsdL': 1,
            'neur_coords': None,
            'neur_grid': None,
            'coord_modif': coord_modif,
            'spatial_kernel': 'cos',  
            'wavelet_amp': 1.0,
            'wavelet_steepness': 0.6,
        
            # DBS (RL agent) parameters
            'elec_coords': [stim_rec_locus_coordinates[4][0]],
            'rec_coords': [stim_rec_locus_coordinates[4][1]],
            'directed_stimulation': False,
            'conduct_modifier': 0.1,  # the bigger, the less is kernel
            'recording_kernel': 'gaussian',
        #     'reinit_spatial_params': False,
            'locus_size': locus_size,
            'locus_center': stim_rec_locus_coordinates[4][2],

            # (0.012, 0.988) - irl params 
            'transient_state_len': 200.,  # [units]
            'electrode_width': 0.15,    # [units]
            'electrode_pause': 0.75,    # [units] 
            'electrode_amps':  [0.],   ## this is RL action   # V
            'dbs_action_bounds': [-5, 5], 

            'electrode_prc_scaling': 1.0,    #
            'electrode_prc_type': 'dummy',
            'naive_dbs': False,
        
            # Stimulation parameters
            'verbose_dt': 0.05,  # [units]
            'total_episode_len': 1000,  # [units],
            'reward_func': None,
            'observe_wind_counts': 130,  # [counts], how many periods we show to RL agent in sliding window

            'init_state_type': 'normal',
            'init_state_mean': np.pi,
            'init_state_sd': 0.6,

            # Temporal features parameters
            'temporal_drift': False,
            'random_freq_update': True,
            'save_events': False,  # to save times of temporal drift
            'electrode_drift_freq': 0,
            'plasticity_drift_freq': 0,
            'plasticity_percent': 0,   # [%]
            'reset_plasticity_episode': 0,

            'encapsulation_drift_freq': 0,
            'encapsulation_percent': 0,    # [%]
            'mov_modulation_drift_freq': 0,

            # Spatial features parameters
            'spatial_feature': False,
            'spatial_var_freq': 0, # for train, how freq-ly we vary electrode coordinates  and RAND_SEED
    }

eval_envs_list = [eval0, eval1, eval2, eval3, eval4]
checking = 'env1'