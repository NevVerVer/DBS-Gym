import os 
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from environment.utils import generate_w0_with_locus
from aDBS_RL.evaluate_HF_DBS import make_env, evaluate_hf_dbs

# We import env version 
from environment.env_configs.env1 import (
    n_neurons, grid_size, coord_modif,
    eval_envs_list, checking
)
np.random.seed(228)


if __name__ == "__main__":
    DIR = 'data/rl_agents_trained_env1'

    # List all agents 
    all_models = os.listdir(DIR)

    for model_type in ['ppo', 'sac', 'ddpg']:
        for rew in ['R1', 'R2', 'R3']:

            # Choose model and rewrds from parsed name of model
            model_name = f'env1_{rew}_{model_type}'
            # Find existing name 
            file_idx = np.asarray([int(model_name in m) for m in all_models])
            idx = np.argwhere(file_idx==1)[0][0]
            model_path = os.path.join(DIR, all_models[idx])
            print('LOADING:   ', all_models[idx])
            print('++'*30)

            # Load the model
            # tb_log = "tensorboard_log/"
            if model_type == 'ppo':
                model = PPO.load(model_path)
            elif model_type == 'sac':
                model = SAC.load(model_path)
            elif model_type == 'ddpg':
                model = DDPG.load(model_path)
            
            # Set parameters for validation 
            NUMBER_OF_ENV = 5
            each_env_run_episodes = 2

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

            # 3. Print all info to not mistake
            print('^^'*30)
            print('THE ENVIROMENT IS:', checking)
            print('NUMBER OF ENVS:', NUMBER_OF_ENV)
            print('Each env with run for: ', each_env_run_episodes, ' times')

            # print('MODEL PREDICT: ', model.predict(np.ones(5)))
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
            true_energy = e_mean * envs_cpu[0].params_dict['dbs_action_bounds'][1] / each_env_run_episodes
            true_energy_sd = e_sd * envs_cpu[0].params_dict['dbs_action_bounds'][1] / each_env_run_episodes
            print('Energy is = ', true_energy, ' sd= ', true_energy_sd)

            # Save all to file
            with open('data/eval_results.json', 'a') as f:
                res = {'env':checking,
                        'agent': f'{model_type}_{rew}', 
                        'bbpow mean':bbpow_mean,
                        'bbpow sd':bbpow_sd, 
                        'energy mean':true_energy,
                        'energy sd':true_energy_sd,
                        }
                f.write(str(res) + '\n')
                f.close()
