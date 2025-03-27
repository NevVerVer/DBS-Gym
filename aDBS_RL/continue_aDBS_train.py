import argparse
import numpy as np
import os
import gym
import shutil
from neurokuramoto.model_v1 import SpatialKuramoto
from neurokuramoto.utils import generate_w0_with_locus
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from neurokuramoto.custom_callbacks import TensorboardCallback, EvalCallback_
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# from data.configs.env0 import (
    # n_neurons, grid_size, coord_modif,
    # params_dict_train, eval_envs_list,
# )
from neurokuramoto.simple_agents import HFDBS

from data.configs.env2 import (    # !!!!!!!! NOTE
    n_neurons, grid_size, coord_modif,
    params_dict_train, eval_envs_list,
)


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--exp_name', type=str, help='experiment name for tb logging')
    parser.add_argument('-a', '--agent', type=str, help='experiment name for tb logging')
    parser.add_argument('-mn', '--model_name', type=str, help='model name for saving')
    parser.add_argument('-ml', '--model_load_name', type=str, help='model name for loading')

    parser.add_argument('-R', '--reward', type=str, help='what reward to use?')
    parser.add_argument('-N', '--num_steps', type=int, default=100_000, help='train steps num')
    parser.add_argument('-rs', '--random_seed', type=int, default=228, help='generator random seed')
    parser.add_argument('-save', '--save_freq', type=int, default=50_000, help='model saving perioud')

    parser.add_argument('-no_eval', '--no_eval', action='store_true', help='turn off eval?')    # type=bool, 
    parser.add_argument('-eval_freq', '--eval_freq', type=int, default=20_000, help='model eval perioud')
    parser.add_argument('-evalnum', '--eval_episodes_num', type=int, default=4, help='how many episodes to run')
    parser.add_argument('-env_eval', '--env_eval_num', type=int, default=4, help='Number of envs for evaluation')
    args = parser.parse_args()
    return args


def make_env(d):
    """
    Creates environment for eval
    """
    def _init():
        env = Monitor(SpatialKuramoto(params_dict=d),
                      filename=None)
        return env
    return _init


if __name__ == "__main__":

    args = parsing_args()
    # np.random.seed(args.random_seed)

    # Load model params
    # p = f'data/configs/{args.config_path}'
    # TODO: in future

    # Create directory for experiment (for imgs, csv, model.tf, etc.)
    MAIN_DIR = 'data/validation_results'
    dir = os.path.join(MAIN_DIR, args.exp_name, "_CONTINUE_TRAIN")

    if os.path.exists(dir): 
        shutil.rmtree(dir)
        print('Folder existed. Deleted it!', '--'*40)

    os.makedirs(dir) 
    save_models_dir = os.path.join(dir, 'saved_models')  # for saving model
    os.makedirs(save_models_dir) 
    eval_dir = os.path.join(dir, 'eval_results') # for saving evaluation results
    os.makedirs(eval_dir) 

    # Create csv for the experiment logging
    csv_name = os.path.join(dir, f'{args.exp_name}_stats.csv')
    if os.path.isfile(csv_name):
        raise FileExistsError(f'{csv_name} already exist! Use other name')
    else:
        with open(csv_name, "w") as f:
            pass

    # Create folder to save temporal events
    params_dict_train['log_path'] = os.path.join(dir, 'train_events')
    if params_dict_train['save_events'] and params_dict_train['log_path'] is not None:
        os.makedirs(params_dict_train['log_path'], exist_ok=True)

    # Define natural freqs. of neurons 
    (w0, ncoords, ngrid,
     w0_temp_, w_locus, lmask) = generate_w0_with_locus(
                n_neurons, grid_size, coord_modif,
                locus_center=params_dict_train['locus_center'],
                locus_size=params_dict_train['locus_size'],
                wmuL=params_dict_train['wmuL'],
                wsdL=params_dict_train['wsdL'],
                show=False, vertical_layer=4)

    # 1. Instantiate the env and check
    params_dict_train['w0'] = w0
    params_dict_train['w0_without_locus'] = w0_temp_
    params_dict_train['locus_without_w0'] = w_locus
    params_dict_train['locus_mask'] = lmask

    params_dict_train['neur_coords'] = ncoords
    params_dict_train['neur_grid'] = ngrid
    params_dict_train['reward_func'] = args.reward

    env = SpatialKuramoto(params_dict=params_dict_train)

    # check_env(env)
    # print('Checked env! Starting train')
    # _= env.reset(save_init=True)

    # 2. Define and Train online agent
    tb_log = "tensorboard_log/"
    if args.agent == 'PPO':
        model = PPO.load(args.model_load_name)
    elif args.agent == 'SAC':
        model = SAC.load(args.model_load_name)
    elif args.agent == 'HFDBS':
        model = HFDBS(env, verbose=0, tensorboard_log=tb_log,
                      action=5.)
    elif args.agent == 'DDPG':
        model = DDPG.load(args.model_load_name)
    
    # Add env to countinue training
    model.set_env(env)

    # 3. Define callbacks
    checkpoint_callback = CheckpointCallback(
                            save_freq=args.save_freq,
                            save_path=save_models_dir,
                            name_prefix=args.model_name,
                            save_replay_buffer=False,
                            save_vecnormalize=False,)
    
    main_callback = TensorboardCallback(csv_name)

    # 4. Instantiate environments for evaluation
    if args.no_eval:
        callback_list = CallbackList([main_callback, checkpoint_callback])

    else:
        print('***Instantiate environments for evaluation***')
        eval_envs_list_new = []
        for n_env in range(args.env_eval_num):
            eval_d = eval_envs_list[n_env]
            eval_d['neur_coords'] = ncoords
            eval_d['neur_grid'] = ngrid
            eval_d['reward_func'] = args.reward
            (w0_eval, _, _, w0_temp_eval,
            w_locus_eval, lmask_eval) = generate_w0_with_locus(
                        n_neurons, grid_size,
                        coord_modif,
                        locus_center=eval_d['locus_center'],
                        locus_size=eval_d['locus_size'] ,
                        wmuL=17, wsdL=1)
            eval_d['w0'] = w0_eval
            eval_d['w0_without_locus'] = w0_temp_eval
            eval_d['locus_without_w0'] = w_locus_eval
            eval_d['locus_mask'] = lmask_eval

            eval_envs_list_new.append(make_env(eval_d))

        # envs_cpu = SubprocVecEnv(eval_envs_list_new)      
        envs_cpu = DummyVecEnv(eval_envs_list_new)

        eval_callback = EvalCallback_(envs_cpu,
                        n_eval_episodes=args.eval_episodes_num,   # ????
                        verbose=1, best_model_save_path=save_models_dir,
                        log_path=eval_dir, eval_freq=args.eval_freq,
                        deterministic=True, render=False, warn=False)
        
        callback_list = CallbackList([ main_callback, checkpoint_callback,
                                    eval_callback ])
    # 5. Train
    model.learn(total_timesteps=args.num_steps,
                callback=callback_list,      # !!!!!!!!!!!!!!!!!!!!!! NOTE
                log_interval=100,  # it is "update intervals" at each n_steps
                tb_log_name=args.exp_name,
                progress_bar=True,
                reset_num_timesteps=False
                )
    # 6. Save model
    msave_name = os.path.join(save_models_dir, f'{args.model_name}_{args.num_steps}_FIN_CONT.tf')
    model.save(msave_name)
    env.close()
    print('Successfully done!!!!')