# utils for visualisation and pre-processing
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
import seaborn as sns 
import logging
import os
from tqdm import tqdm
import imageio
import shutil

from scipy.integrate import quad
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable

PATH = '/home/ekuzmina/brainlearning/'
DPI = 300
SHOW = True


def calc_beta_band_power(sig, dt, beta_a, beta_b):
    n = sig.shape[0]
    ft = np.abs(np.fft.rfft(sig) / n)**2 * 2
    freq = np.fft.rfftfreq(n, dt)
    idx = np.where((freq > beta_a) & (freq < beta_b))
    band_power = np.sum(ft[idx])
    return band_power


def spherical_coordinates(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Returns azimuth (theta), elevation (phi), and radius (r).
    """
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-5
    theta = np.arctan2(y, x)  # azimuthal angle
    phi = 0 # np.arccos(z / r)  # polar angle, NOTE: D=[-1, 1]
    return theta, phi, r


def create_directed_stim_masks(grid_points, center, center_idx):
    """
    Divides the grid of points into 3 parts, separated by 120 degrees in azimuthal angle.
    Returns 3 masks for each part.
    """
    x, y, z = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]
    x_shifted = x - center[0]
    y_shifted = y - center[1]
    z_shifted = z - center[2]

    theta, phi, r = spherical_coordinates(x_shifted, y_shifted, z_shifted)
    mask1 = (theta >= -np.pi/3) & (theta < np.pi/3)
    mask2 = (theta >= np.pi/3) & (theta <= np.pi)
    mask3 = (theta >= -np.pi) & (theta < -np.pi/3)
    for m in [mask1, mask2, mask3]:
        m[center_idx] = True
    return mask1, mask2, mask3


def plot_psd(sig, dt, cut_idx, log_scale=False):
    n = sig.shape[0]
    ft = np.abs(np.fft.rfft(sig)/n)**2 * 2
    # ft = ft / n * 2  # remove for power calculation
    freq = np.fft.rfftfreq(n, dt)
    
    if log_scale:
        plt.semilogy(freq[:cut_idx], ft[:cut_idx])
    else:
        plt.plot(freq[:cut_idx], ft[:cut_idx])
    plt.xlabel('Hz')
    plt.ylabel('Volt**2/hz')
    plt.axvline(12.5, color='red')
    plt.axvline(33.5, color='red')
    plt.title(f'Power Spectral Density, log_scale={log_scale}')
    plt.show()


def start_logger(lname='l', log_file='logfile.log'):
    logger = logging.getLogger(lname)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create a console handler and set the level to DEBUG
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a file handler and set the level to DEBUG
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

### =========================================================
# Phase Coherence Functions

def rad_sec2herz(x):
    return x / (2*np.pi)


def circular_mean(angles):
    """
    angles: [rad]
    """
    sum_sin = np.sum(np.sin(angles))
    sum_cos = np.sum(np.cos(angles))
    mean_angle = np.arctan2(sum_sin, sum_cos)
    
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    
    return mean_angle


def calculate_phase_coherence(angles):
    """
    angles: list of phases [rad]
    returns complex coherence number
    """
    complex_phases = np.exp(1j * angles)
    mean_complex_phase = np.mean(complex_phases)

    coherence = np.abs(mean_complex_phase)
    return coherence, mean_complex_phase


def plot_polar_distibution(radians, colors, save=False):
    """
    radians: neuron phases [rad]
    colors: matrix or vector of values to assign colors
    """
    if len(colors.shape) > 1:
        colors = reshape_grid2vec(colors)

    flat_colors = colors
    colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    colors_map = plt.cm.copper(colors)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i, rad in enumerate(radians):
        pr = ax.plot(rad, 1., c=colors_map[i], marker='o', markersize=8)

    sm = plt.cm.ScalarMappable(cmap='copper', norm=plt.Normalize(vmin=np.min(flat_colors),
                                                               vmax=np.max(flat_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Natural Frequency', rotation=270, labelpad=15)

    r, mean_comp_phase = calculate_phase_coherence(radians)
    centroid_angle = np.angle(mean_comp_phase)
    ax.plot([0, centroid_angle], [0, r], '--', c='black', alpha=0.8)
    ax.plot(centroid_angle, r, 'o', markersize=11, c='black')

    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)
    ax.set_rlim(0, 1.2)
    if save:
        return fig
    else:
        plt.show()

### =========================================================
# Working with folders and gif generation

def delete_folder(folder_path, logger=None):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        if logger:
            logger.info(f"Deleted folder: {folder_path}")
        else:
            print(f"Deleted folder: {folder_path}")
    else:
        if logger:
            logger.info(f"The specified path {folder_path} does not exist or is not a directory.")
        else:
            print(f"The specified path {folder_path} does not exist or is not a directory.")


def prepare_dir_to_frames(folder_name, logger=None):
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Check if the folder is empty
        if not os.listdir(folder_name):
            if logger:
                logger.info(f"The folder '{folder_name}' exists and is empty.")
            else:
                print(f"The folder '{folder_name}' exists and is empty.")
        else:
            # If the folder contains files, delete them
            try:
                for filename in os.listdir(folder_name):
                    file_path = os.path.join(folder_name, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                if logger:
                    logger.info(f"The folder '{folder_name}' existed with files, which have been deleted.")
                else:
                    print(f"The folder '{folder_name}' existed with files, which have been deleted.")
            except Exception as e:
                if logger:
                    logger.info(f"Error deleting files in '{folder_name}': {str(e)}")
                else:
                    print(f"Error deleting files in '{folder_name}': {str(e)}")
    else:
        # If the folder doesn't exist, create it
        try:
            os.mkdir(folder_name)
            if logger:
                logger.info(f"The folder '{folder_name}' has been created.")
            else:
                print(f"The folder '{folder_name}' has been created.")
        except Exception as e:
            if logger:
                logger.info(f"Error creating folder '{folder_name}': {str(e)}")
            else:
                print(f"Error creating folder '{folder_name}': {str(e)}")


def generate_image_name(idx, gif_dir):
    if idx < 0 or idx > 9999:
        raise ValueError("idx must be between 0 and 9999")
    n = f"{idx:04d}"
    return os.path.join(gif_dir, n + '.png')


def render_gif(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    keys = [int(image[:-4]) for image in images]
    xx = np.argsort(keys)
    images = np.array(images)[xx]

    with imageio.get_writer(video_name, mode='I') as writer:
        for filename in images:
            image = imageio.imread(image_folder + "/" + filename)
            writer.append_data(image)


def save_phase_grid_frames(sig, times, electrode_idx, electrode_bool,
                           gif_dir, vmax=6.28, vmin=0, normalize=True,
                           cmap_type='twilight', logger=None):
    """
        sig: shape [grid_size[0], grid_size[1], timesteps]
        times: in msec
        electrode_bool: does dbs works in this time point
    """
    if normalize:
        sig = sig % (2 * np.pi)
    
    if cmap_type == 'husl':  # a little bit of hardcoding :) 
        colorm = sns.color_palette("husl", as_cmap=True)
    elif cmap_type == 'twilight':
        colorm = 'twilight'

    for frame_idx in tqdm(np.arange(0, sig.shape[0])):
        plt.imshow(sig[frame_idx], cmap=colorm, vmax=vmax, vmin=vmin);
        plt.colorbar()

        dbs_status = 'ON' if electrode_bool[frame_idx] else 'OFF'
        dbs_col = 'black' if electrode_bool[frame_idx] else 'gray'
        plt.scatter(electrode_idx[0], electrode_idx[1], c=dbs_col,
                    marker='X', s=260, alpha=1.)
        plt.title(f'Phases at tstep {round(times[frame_idx], 1)}, dbs: {dbs_status}')

        img_name = generate_image_name(frame_idx, gif_dir)
        plt.savefig(img_name, dpi=DPI)
        plt.clf()
        plt.close()
    if logger:
        logger.info(f'Done! {frame_idx} frames saved in folder: {gif_dir}')
    else:
        print(f'Done! {frame_idx} frames saved in folder: {gif_dir}')


def save_phase_distr_frames(sig, times, electrode_idx, electrode_bool,
                            colors_matrix, gif_dir, normalize=True,
                            logger=None):
    """
        sig: shape [grid_size[0], grid_size[1], timesteps]
        times: in msec
        electrode_bool: does dbs works in this time point
    """
    if normalize:
        sig = sig % (2 * np.pi)

    for frame_idx in tqdm(np.arange(0, sig.shape[0])):

        p = plot_polar_distibution(sig[frame_idx], colors_matrix, save=True)

        dbs_status = 'ON' if electrode_bool[frame_idx] else 'OFF'
        phase_coh, _ = calculate_phase_coherence(sig[frame_idx])
        plt.title(f'Phases distr, t_idx {round(times[frame_idx], 1)}, dbs {dbs_status}, coherence={round(phase_coh, 3)}')

        img_name = generate_image_name(frame_idx, gif_dir)
        plt.savefig(img_name, dpi=DPI)
        plt.clf() 
        plt.close()
    if logger:
        logger.info(f'Done! {frame_idx} frames saved in folder {gif_dir}')
    else:
        print(f'Done! {frame_idx} frames saved in folder {gif_dir}')


def save_mean_phase_plot_frames(sig, times, electrode_bool,
                                gif_dir, normalize=True,
                                logger=None):
    if normalize:
        sig = sig % (2 * np.pi)

    x = np.mean(np.cos(sig), axis=1)
    y = np.mean(np.sin(sig), axis=1)
    for frame_idx in tqdm(np.arange(1, sig.shape[0])):

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        plt.plot(x[0:frame_idx], y[0:frame_idx],
                    c='grey', alpha=0.8, zorder=1)
        plt.scatter(x[0:frame_idx], y[0:frame_idx],
                    c=times[0:frame_idx], alpha=0.8,
                     s=30, zorder=2)

        dbs_col = 'red' if electrode_bool[frame_idx] else 'gray' 
        plt.scatter(x[frame_idx], y[frame_idx], s=70,
                    c=dbs_col, zorder=3)
        plt.colorbar()
        plt.axvline(0.0, color='gray', alpha=0.6)
        plt.axhline(0.0, color='gray', alpha=0.6)
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])
        dbs_status = 'ON' if electrode_bool[frame_idx] else 'OFF'
        plt.title(f'Mean phase plot, t: {round(times[frame_idx], 1)}, dbs {dbs_status}')
        
        img_name = generate_image_name(frame_idx, gif_dir)
        plt.savefig(img_name, dpi=DPI)
        plt.clf() 
        plt.close()
    if logger:
        logger.info(f'Done! {frame_idx} frames saved in folder {gif_dir}')
    else:
        print(f'Done! {frame_idx} frames saved in folder {gif_dir}')


def solution2gif(solution, t_eval, total_frames, model,
                 gif_name, dir_name, plot_type, logger=None):
    """
    dir_name: directory to save frames
    gif_name: full path to save gif, with extension ".gif"
    """

    fstep = int(solution.shape[0] / total_frames)
    if (solution.shape[0] / fstep) > 1500:
        raise ValueError("Too many frames. No more than 1500 frames possible!")
    if logger:
        logger.info('step: %s', fstep)
    else: 
        print('step: ', fstep) #fstep * model.dt)

    elec_coord = model.dbs.elec_coord
    y_dot = model.grid_size[1] - elec_coord[1] - 1
    elec_idx_for_plotting = [elec_coord[0], y_dot] 

    # take the times of electrode working, bool array
    electrode_pulse = [model.dbs.make_pulse(i, solution[idx, :]) for idx, i in enumerate(t_eval)]
    electrode_pulse = np.asarray(electrode_pulse, dtype=bool)[:, model.dbs.elec_idx]

    sol, reshaped_sol = [], []
    times, electrode_bool = [], []
    for frame in range(0, solution.shape[0], fstep):
        sol.append(solution[frame])
        reshaped_sol.append(reshape_vec2grid(solution[frame],
                                             coords=model.neur_grid,
                                             grid_size=model.grid_size))
        times.append(t_eval[frame])
        electrode_bool.append(electrode_pulse[frame])

    prepare_dir_to_frames(os.path.join(PATH, dir_name), logger=logger)

    if plot_type == 'phase_grid':
        save_phase_grid_frames(np.asarray(reshaped_sol), times,
                            elec_idx_for_plotting, electrode_bool,
                            gif_dir=os.path.join(PATH, dir_name),
                            vmax=2*np.pi, vmin=0, logger=logger)
        
    elif plot_type == 'phase_dist':
        colors_matrix = model.w0
        save_phase_distr_frames(np.asarray(sol), times,
                                elec_idx_for_plotting, electrode_bool,
                                colors_matrix,
                                gif_dir=os.path.join(PATH, dir_name),
                                normalize=True, logger=logger)
    elif plot_type == 'phase_plot':
        save_mean_phase_plot_frames(np.asarray(sol), times, electrode_bool,
                                   gif_dir=os.path.join(PATH, dir_name),
                                    normalize=True, logger=logger)

    else:
        raise ValueError(f'Wrong plot type: {plot_type}')
    
    render_gif(image_folder=os.path.join(PATH, dir_name),
               video_name=os.path.join(PATH, gif_name))
    if logger:
        logger.info(f'as {gif_name}')
    else:
        print(f'as {gif_name}')

### ==============================================================
# For Kuramoto models

def convers_coord(x, y, grid_size):
    y_dot = grid_size[1] - 1 - y
    return y_dot, x


def reshape_vec2grid(vec, coords, grid_size):
    n = vec.shape[0]
    reshaped_vec = np.zeros((grid_size[1], grid_size[0]))
    for j in range(n):
        x, y = coords[j]
        y_dot = grid_size[1] - y - 1
        reshaped_vec[y_dot, x] = vec[j]
    return reshaped_vec


def reshape_grid2vec(grid):
    grid_size = grid.shape
    vec = []
    for y in np.arange(grid_size[0]):
        for x in np.arange(grid_size[1], 0, -1):
            vec.append(grid[x, y])
    return vec


def generate_neuron_grid(greed_size_x, greed_size_y, n_neurons, 
                         coord_modif=0.1, shuffle=False):
    if n_neurons > greed_size_x * greed_size_y:
        raise ValueError("Number of neurons should be less than grid size.")
    
    x_coords = np.arange(0, greed_size_x)
    y_coords = np.arange(0, greed_size_y)    

    grid_coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    if shuffle:
        np.random.shuffle(grid_coords)

    n_grid_coords = grid_coords[:n_neurons]
    neur_coords = n_grid_coords * coord_modif
    
    return neur_coords, n_grid_coords


def create_distance_matrix(neur_coords):
    n_neurons = neur_coords.shape[0]
    distance_matrix = np.zeros((n_neurons, n_neurons))
    
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            dist = np.linalg.norm(neur_coords[i] - neur_coords[j])
            distance_matrix[i, j] = dist 
            distance_matrix[j, i] = dist
    return distance_matrix


def wavelet_kernel_matrix(distances, amplitude, steepness):
    W = (
        amplitude * (-steepness) *
        (12 * steepness**4 * distances**2 - 8 * steepness**2) *
        np.exp(-steepness * distances**2) / (2 * np.pi)
        )
    return W


def generate_neuron_grid_3D(greed_size_x, 
                         greed_size_y, 
                         greed_size_z,
                         n_neurons, 
                         coord_modif=0.1, shuffle=False):
    if n_neurons > greed_size_x * greed_size_y * greed_size_z:
        raise ValueError("Number of neurons should be less than grid size.")
    
    x_coords = np.arange(0, greed_size_x)
    y_coords = np.arange(0, greed_size_y)  
    z_coords = np.arange(0, greed_size_z)   

    grid_coords = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    if shuffle:
        np.random.shuffle(grid_coords)

    n_grid_coords = grid_coords[:n_neurons]
    neur_coords = n_grid_coords * coord_modif
    
    return neur_coords, n_grid_coords


def wavelet_kernel_matrix(distances, amplitude, steepness):
    W = (
        amplitude * (-steepness) *
        (12 * steepness**4 * distances**2 - 8 * steepness**2) *
        np.exp(-steepness * distances**2) / (2 * np.pi)
        )
    return W


### ===================================================================
# Visualization functions

def plot_phase_wrt_time(sol, ts, normalize=True, tstart=0, tend=5000,
                        save_name=None):
    n = sol.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    if normalize:
        cycle_rad = sol[tstart:tend,:n] % (2 * np.pi)
    else:
        cycle_rad = sol[tstart:tend,:n]
    for nidx in range(n):
        plt.scatter(ts[tstart:tend], cycle_rad[:, nidx], s=0.1, alpha=0.5)
    plt.axhline(3.14, c='black', lw=3., alpha=0.3)
    plt.title("Phase of Each Neuron wrt Time")
    plt.xlabel("Time, ms")
    if normalize:
        plt.ylabel("theta, [cycled_rad]")
    else:
        plt.ylabel("theta, rad")

    if save_name:
        save_name = os.path.join(PATH, save_name)
        plt.savefig(save_name, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    else:
        plt.clf()
        plt.close()

def extract_dbs_times(t_eval, model):
    dbs_times = []
    dbs_pause = model.dbs.pause
    dbs_width = model.dbs.width

    t = model.dbs.start_time
    # print(t, 'dbs_width: ', dbs_width, ' pause: ', dbs_pause)
    while t <= t_eval[-1]:
        dbs_times.append([t, t+dbs_width])
        t += dbs_width
        t += dbs_pause
    return np.asarray(dbs_times)


def plot_xy_wrt_time(sol, t_eval, model, 
                     plot_dbs=False, plot_y=False,
                     tstart=0, tend=None,
                     nstart=0, nend=None,
                     real_units_step=1,
                     save_name=None, logger=None):
    if not tend:
        tend = sol.shape[0]
    if not nend:
        nend = np.min([50, sol.shape[1]])
    
    x = np.sin(sol) if plot_y else np.cos(sol)
    t_eval = t_eval / real_units_step
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    ax.plot(t_eval[tstart:tend], x[tstart:tend, nstart:nend])
    name = 'y (sin)' if plot_y else 'x (cos)'
    ax.set_title(f'{name} w.r.t time (from {nstart} to {nend} neurons), from {t_eval[tstart]} to {t_eval[tend]}')
    ax.set_xlim([t_eval[tstart], t_eval[tend]])

    if plot_dbs:
        dbs_times = extract_dbs_times(t_eval, model)
        times_in_range = np.where((dbs_times >= t_eval[tstart]) & (dbs_times <= t_eval[tend]))[0]
        if len(times_in_range) == 0:
            if logger:
                logger.info('No DBS!')
            else:
                print('No DBS!')
        else:
            times_in_range = np.unique(times_in_range)
            dbs_times = dbs_times[times_in_range]
            if dbs_times[0, 0] < t_eval[tstart]:
                dbs_times[0, 0] = t_eval[tstart]
            if dbs_times[-1, 1] > t_eval[tend]:
                dbs_times[-1, 1] = t_eval[tend]

            for dbs_t in dbs_times:
                ax.axvspan(dbs_t[0], dbs_t[1], alpha=0.3, color='grey')
                ax.axvline(dbs_t[0], color='black', alpha=0.3)

    if save_name:
        save_name = os.path.join(PATH, save_name)
        plt.savefig(save_name, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_mean_field(sol, t_eval, model, plot_dbs=False,
                    plot_y=False,
                    tstart=0, tend=None,
                    real_units_step=1,
                    save_name=None, logger=None):
    if not tend:
        tend = sol.shape[0]

    x = np.sin(sol) if plot_y else np.cos(sol)
    xmean = np.mean(x, axis=1)
    t_eval = t_eval / real_units_step
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    name = 'y (sin)' if plot_y else 'x (cos)'
    ax.plot(t_eval[tstart:tend], xmean[tstart:tend],
               label=f'{name} mean field')
    ax.legend()
    ax.set_title(f'Mean field for {name}')

    if plot_dbs:
        dbs_times = extract_dbs_times(t_eval, model)
        times_in_range = np.where((dbs_times >= t_eval[tstart]) & (dbs_times <= t_eval[tend]))[0]
        if len(times_in_range) == 0:
            if logger:
                logger.info('No DBS!')
            else:
                print('No DBS!')
        else:
            times_in_range = np.unique(times_in_range)
            dbs_times = dbs_times[times_in_range]
            if dbs_times[0, 0] < t_eval[tstart]:
                dbs_times[0, 0] = t_eval[tstart]
            if dbs_times[-1, 1] > t_eval[tend]:
                dbs_times[-1, 1] = t_eval[tend]

            for dbs_t in dbs_times:
                ax.axvspan(dbs_t[0], dbs_t[1], alpha=0.3, color='grey')
                ax.axvline(dbs_t[0], color='black', alpha=0.3)

    ax.set_xlim([t_eval[tstart], t_eval[tend]])
    ax.axhline(0.0, color='grey', alpha=0.6)
    ax.axhline(0.5, color='grey', alpha=0.4)
    ax.axhline(-0.5, color='grey', alpha=0.4)
    ax.axhline(1.0, color='grey', alpha=0.4)
    ax.axhline(-1.0, color='grey', alpha=0.4)
    if save_name:
        save_name = os.path.join(PATH, save_name)
        plt.savefig(save_name, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_phase_coherence_wrt_time(sol, t_eval, 
                                  model, plot_dbs=False,
                                  tstart=0, tend=None,
                                  nstart=0, nend=None,
                                  real_units_step=1, 
                                  save_name=None, logger=None):
    t_eval = t_eval / real_units_step
    sol = sol[tstart:tend, nstart:nend]
    phase_coh_wrt_time = []
    for t in range(sol.shape[0]):
        phase_coh, _ = calculate_phase_coherence(sol[t])
        phase_coh_wrt_time.append(phase_coh)

    fig, ax = plt.subplots(1,1, figsize=(12, 3))
    if plot_dbs:
        dbs_times = extract_dbs_times(t_eval, model)
        times_in_range = np.where((dbs_times >= t_eval[tstart]) & (dbs_times <= t_eval[tend]))[0]
        if len(times_in_range) == 0:
            if logger:
                logger.info('No DBS!')
            else:
                print('No DBS!')
        else:
            times_in_range = np.unique(times_in_range)
            dbs_times = dbs_times[times_in_range]
            if dbs_times[0, 0] < t_eval[tstart]:
                dbs_times[0, 0] = t_eval[tstart]
            if dbs_times[-1, 1] > t_eval[tend]:
                dbs_times[-1, 1] = t_eval[tend]

            for dbs_t in dbs_times:
                ax.axvspan(dbs_t[0], dbs_t[1], alpha=0.3, color='grey')
                ax.axvline(dbs_t[0], color='black', alpha=0.3)

    ax.plot(t_eval[tstart:tend], phase_coh_wrt_time)
    ax.axhline(0.0, color='gray', alpha=0.9)
    ax.set_title(f'Phase Coherence w.r.t. time, for {nstart}-{nend} neurons')
    
    if save_name:
        save_name = os.path.join(PATH, save_name)
        plt.savefig(save_name, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_mean_phase_plot(sol, t_eval, tstart=0, tend=None,
                         real_units_step=1, 
                         save_name=None):
    if not tend:
        tend = sol.shape[0]
    x = np.mean(np.cos(sol), axis=1)
    y = np.mean(np.sin(sol), axis=1)
    t_eval = t_eval / real_units_step
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    plt.scatter(x[tstart:tend], y[tstart:tend], c=t_eval[tstart:tend], alpha=0.8, s=1)
    plt.colorbar()
    if save_name:
        save_name = os.path.join(PATH, save_name)
        plt.savefig(save_name, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_natural_freqs(w0, neur_grid, grid_size, save_name=None):
    rw0 = reshape_vec2grid(vec=w0,
                           coords=neur_grid,
                           grid_size=grid_size)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(rw0, cmap='twilight', vmin=0, vmax=2*np.pi)
    for i in range(rw0.shape[0]):
        for j in range(rw0.shape[1]):
            ax.text(j, i, f'{rw0[i, j]:.2f}', ha='center',
                    va='center', color='black')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Values of initial natural frequencies, [rad/sec]')

    if save_name:
        save_name = os.path.join(PATH, save_name)
        plt.savefig(save_name, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_neurons_and_dbs(solution, dbs_sig, t_eval, model, 
                         ns, ne, tstart, tend, real_units_step=1,
                         save_name=None):

    fig, ax = plt.subplots(ne-ns+1, 1, figsize=(12, 10))
    sol = np.cos(np.asarray(solution))  # to cartesian
    t_eval = t_eval / real_units_step
    
    jj = 0
    for i in range(ns, ne):
        ax[jj].plot(t_eval[tstart:tend], sol[tstart:tend, i], label=f'neuron {i}') 
        ax[jj].axhline(1, color='grey', alpha=0.5)
        ax[jj].axhline(0, color='grey', alpha=0.3)
        ax[jj].axhline(-1, color='grey', alpha=0.5)
        ax[jj].axvline(model.dbs.start_time, color='black', alpha=0.8)

        ax[jj].plot(t_eval[tstart:tend], dbs_sig[tstart:tend, i], 
                    label=f'conductance: {round(model.dbs.conductance[i], 2)}', 
                    color='red', alpha=0.4)
        ax[jj].legend()
        jj += 1

    ax[jj].plot(t_eval[tstart:tend], np.mean(sol[tstart:tend, ns:ne], axis=1),
                label=f'{ns}-{ne} neurons mean field')
    ax[jj].plot(t_eval[tstart:tend], np.mean(sol[tstart:tend], axis=1),
                label='Total mean field',
                color='red')
    ax[jj].axhline(0, color='grey', alpha=0.3)
    ax[jj].axvline(model.dbs.start_time, color='black', alpha=0.7)
    ax[jj].legend()
    ax[0].set_title(f'Neural and DBS Activity (neurons {ns}-{ne})', fontsize=13)

    if save_name:
        save_name = os.path.join(PATH, save_name)
        plt.savefig(save_name, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    else:
        plt.clf()
        plt.close()


def band_pass_envelope(signal, fs, lowcut=12, highcut=30, order=5):
    """
    Apply a band-pass filter to the input signal and calculate its envelope.

    Parameters:
    - signal: The input signal (1D numpy array).
    - fs: The sampling rate of the signal (Hz).
    - lowcut: The lower cutoff frequency of the band-pass filter (Hz).
    - highcut: The higher cutoff frequency of the band-pass filter (Hz).

    Returns:
    - filtered_signal: The signal after applying the band-pass filter.
    - envelope: The envelope of the filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered_signal))

    return filtered_signal, envelope


def remove_negative_w0(w0):
    idx = np.where(w0 <= 0.)[0]
    n = np.random.randn(len(idx)) * 0.05
    w0[idx] = np.abs(n) + np.mean(w0)
    return w0


def sec2units(x):
    units_for_1_sec = 100  # [units]
    return x * units_for_1_sec

def units2sec(x):
    units_for_1_sec = 100  # [units]
    return x / units_for_1_sec


def calc_envelope(sig):
    return np.abs(hilbert(sig))

# def normalize(t, y):
#     exceeded_indices = np.where(y > 2*np.pi)[0]
#     if len(exceeded_indices) > 0:
#         y[exceeded_indices] = np.fmod(y[exceeded_indices], 2*np.pi)
#     return y


### w0 and locus initialization -----------------------------------------------

def generate_w0_samples(N, lf_peak=6, beta_peak=10, show=False):
    y = [6, 7.7, lf_peak, 7.7, 4, 3.5, 4, 5, 5.7, beta_peak, 5.7, 4.9, 2.3, 1.2, 0.8, 0.75, 0.7, 0.7, 0.68]
    x = [0, 1.8, 2.5, 3.3, 4.5, 5.5, 8, 12.5, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60]
    # y = [6, 6  , lf_peak, 7., 4, 3.5, 4.5, 5.5, 7.7, beta_peak, 7.7, 6.9, 2.3, 1.2, 0.8, 0.75, 0.7, 0.7, 0.68]
    # x = [0, 1.8, 2.5, 3.3, 4.5, 5.5, 8, 12.5, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60]
    degree = 10
    poly = np.poly1d(np.polyfit(x, y, degree))
    x_range = np.linspace(np.min(x), 30, 1000)
    y_poly = poly(x_range)

    # Normalize the polynomial to make it a valid PDF
    def pdf(x):
        return np.maximum(poly(x), 0)  # Ensure non-negative values
    normalization_constant, _ = quad(pdf, np.min(x), np.max(x))
    pdf_normalized = lambda x: pdf(x) / normalization_constant

    cdf_values = np.cumsum(pdf_normalized(x_range))
    cdf_values /= cdf_values[-1]  # Normalize CDF to range from 0 to 1
    # Create an interpolating function for the inverse CDF
    inverse_cdf = interp1d(cdf_values, x_range, bounds_error=False,
                        fill_value=(x_range[0], x_range[-1]))
    random_values = np.random.rand(N)
    samples = inverse_cdf(random_values)

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].scatter(x, y, color='red', label='Data Points')
        ax[1].hist(samples, bins=25, density=True, alpha=0.5, label='Sampled Points')
        ax[1].plot(x_range, pdf_normalized(x_range), label='Normalized PDF')
        for i in [0, 1]:
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('Probability Density')
            ax[i].legend()
        plt.show()

    return samples


def create_oscillation_locus(neur_grid, grid_size, locus_coord, 
                             locus_size):
    l_idx = locus_coord[0] * grid_size[2]**2 + locus_coord[1] * grid_size[1] + locus_coord[2]
    dist_matrix = create_distance_matrix(neur_grid*locus_size)
    dist_vector = dist_matrix[l_idx]
    locus_mask = np.where(1 - dist_vector < 0.0, 0., 1.)
    return locus_mask


def reshape_vec2grid_3D(vec, coords, grid_size):
    reshaped_vec = np.zeros(grid_size)
    for j in range(vec.shape[0]):
        x, y, z = coords[j]
        reshaped_vec[x, y, z] = vec[j]
    return reshaped_vec


def apply_locus_mask(w0, w_locus, lmask):
    lmask_inv = lmask * -1 + 1
    w0_temp = w0 * lmask_inv + w_locus * lmask  # Apply mask to w0
    # w0_temp = w0_temp * 0.065  # from Hz to rad/sec
    return w0_temp


def generate_w0_with_locus(n_neurons, grid_size, coord_modif,
                           locus_center, locus_size, wmuL, wsdL,
                           show=True, vertical_layer=4,):
    """
    Returns w0 in rads!
    """
    # 1. Generate values of w0 without locus, in degrees
    w0_temp_deg = generate_w0_samples(n_neurons, show=show)

    # 2. Generate the spatial grid of neurons 
    neur_coords, neur_grid = generate_neuron_grid_3D(*grid_size,
                                                 n_neurons,
                                                 coord_modif=coord_modif)
    # 3. Generate beta-oscillation locus mask (0, 1 values)
    lmask_bool = create_oscillation_locus(neur_grid, grid_size,
                                        locus_coord=locus_center, 
                                        locus_size=locus_size) # the bigger, the less is mask
    # 4. Generate values for the beta locus, in degrees
    w_locus_deg = np.random.uniform(low=wmuL-wsdL, high=wmuL+wsdL,
                                    size=(n_neurons))
    # 5. Apply locus maks to w0, in deg
    w0_temp_with_locus = apply_locus_mask(w0_temp_deg, w_locus_deg,
                                          lmask_bool)
    # 6. Convert from Hz to rad/sec all values
    w0_temp_with_locus_rad = w0_temp_with_locus * 0.065
    w0_temp_rad = w0_temp_deg * 0.065
    w_locus_rad = w_locus_deg * 0.065

    # 7. Plot the results if needed
    # if show:
        # TODO run plotting func

    return (w0_temp_with_locus_rad, neur_coords, neur_grid, 
            w0_temp_rad, w_locus_rad, lmask_bool)

    # TODO: to separate plotting function  
    # if show:
        # 1. Reshape w0 values for visualization
        # gamma_reshaped = reshape_vec2grid_3D(w0_temp_, neur_grid, grid_size)
    #     fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    #     im2 = axes[0].imshow(gamma_reshaped[:, :, vertical_layer]*0.065,
    #                         interpolation=None)
    #     divider = make_axes_locatable(axes[0])
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im2, cax=cax, orientation='vertical',
    #                 label='Natural frequency of oscillation, rad/sec')
    #     gamma_reshaped = reshape_vec2grid_3D(w0_temp_with_locus, neur_grid, grid_size)
    #     im = axes[1].imshow(gamma_reshaped[:, :, vertical_layer], interpolation=None)
    #     # add colorbar
    #     divider = make_axes_locatable(axes[1])
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im, cax=cax, orientation='vertical',
    #                 label='Natural frequency of oscillation, rad/sec')
    #     axes[2].hist(w0_temp_with_locus, bins=60, alpha=0.5)
    #     axes[2].axvline(1.3, c='black', alpha=0.5)
    #     axes[2].axvline(1.3-0.2, c='black', alpha=0.5)
    #     axes[2].axvline(1.3-0.2*3, c='black', alpha=0.5)
    #     axes[2].axvline(1.3+0.2, c='black', alpha=0.5)
    #     axes[2].axvline(1.3+0.2*3, c='black', alpha=0.5)
    #     axes[2].set_title('w0 + locus')
    #     # Plot Electrode perseption field
    #     lmask_reshaped = reshape_vec2grid_3D(lmask, neur_grid, grid_size)
    #     coords_xy = np.where(lmask_reshaped==1)
    #     lmask_coords = np.asarray([coords_xy[1], coords_xy[0]])
    #     axes[1].scatter(lmask_coords[0], lmask_coords[1],
    #                     c='#0a5e21', marker='o', s=18, alpha=1,
    #                     label='Beta oscillation locus')
    #     # Center of mask  # TODO: fix
    #     y_dot = grid_size[1] - locus_center[1] - 2
    #     axes[1].scatter(locus_center[0], y_dot, c='#0a5e21', marker='X',
    #                     s=100, alpha=0.75)
    #     for i in [0, 1]:
    #         axes[i].set_title(f'Neural grid', fontsize=14)
    #         axes[i].set_xlabel('Neuron index', fontsize=12)
    #         axes[i].set_ylabel('Neuron index', fontsize=12)
    #         axes[i].legend(loc='lower right')
    #     plt.show()

    
# def generate_w0_with_locus(n_neurons, grid_size, coord_modif,
#                            locus_center, locus_size,
#                            wmuL, wsdL, show=False, vertical_layer=4,):
#     w0_temp_ = generate_w0_samples(n_neurons, show=True)
#     neur_coords_temp, neur_grid_temp = generate_neuron_grid_3D(*grid_size,
#                                                  n_neurons,
#                                                  coord_modif=coord_modif)
#     gamma_reshaped = reshape_vec2grid_3D(w0_temp_, neur_grid_temp, grid_size)
#     lmask = create_oscillation_locus(neur_grid_temp, grid_size,
#                                      locus_coord=locus_center, 
#                                      locus_size=locus_size) # the bigger, the less is mask
#     w_locus = np.random.uniform(low=wmuL-wsdL, high=wmuL+wsdL, size=(n_neurons))
#     w0_temp_with_locus = apply_locus_mask(w0_temp_, w_locus, lmask)

#     if show:
#         fig, axes = plt.subplots(1, 3, figsize=(16, 4))
#         im2 = axes[0].imshow(gamma_reshaped[:, :, vertical_layer]*0.065,
#                             interpolation=None)
#         divider = make_axes_locatable(axes[0])
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(im2, cax=cax, orientation='vertical',
#                     label='Natural frequency of oscillation, rad/sec')
#         gamma_reshaped = reshape_vec2grid_3D(w0_temp_with_locus, neur_grid_temp, grid_size)
#         im = axes[1].imshow(gamma_reshaped[:, :, vertical_layer], interpolation=None)
#         # add colorbar
#         divider = make_axes_locatable(axes[1])
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(im, cax=cax, orientation='vertical',
#                     label='Natural frequency of oscillation, rad/sec')
#         axes[2].hist(w0_temp_with_locus, bins=60, alpha=0.5)
#         axes[2].axvline(1.3, c='black', alpha=0.5)
#         axes[2].axvline(1.3-0.2, c='black', alpha=0.5)
#         axes[2].axvline(1.3-0.2*3, c='black', alpha=0.5)
#         axes[2].axvline(1.3+0.2, c='black', alpha=0.5)
#         axes[2].axvline(1.3+0.2*3, c='black', alpha=0.5)
#         axes[2].set_title('w0 + locus')
#         # Plot Electrode perseption field
#         lmask_reshaped = reshape_vec2grid_3D(lmask, neur_grid_temp, grid_size)
#         coords_xy = np.where(lmask_reshaped==1)
#         lmask_coords = np.asarray([coords_xy[1], coords_xy[0]])
#         axes[1].scatter(lmask_coords[0], lmask_coords[1],
#                         c='#0a5e21', marker='o', s=18, alpha=1,
#                         label='Beta oscillation locus')
#         # Center of mask
#         y_dot = grid_size[1] - locus_center[1] - 2
#         axes[1].scatter(locus_center[0], y_dot, c='#0a5e21', marker='X',
#                         s=100, alpha=0.75)
#         for i in [0, 1]:
#             axes[i].set_title(f'Neural grid', fontsize=14)
#             axes[i].set_xlabel('Neuron index', fontsize=12)
#             axes[i].set_ylabel('Neuron index', fontsize=12)
#             axes[i].legend(loc='lower right')
#         plt.show()

#     return (w0_temp_with_locus, neur_coords_temp, neur_grid_temp, 
#             w0_temp_, w_locus, lmask)
    