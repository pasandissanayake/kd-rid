import sys
sys.path.append('..')

import os

import pickle as pkl
from pid.rus import *
from pid.utils import clustering

from dit.pid import iwedge
from dit import Distribution
from dit.shannon import mutual_information

import re
import time
from datetime import timedelta



def reshape_list(input_list, n_reps):
    n_blocks = len(input_list) // n_reps

    reps = []
    for i in range(n_reps):
        a = []
        for j in range(n_blocks):
            a.extend(input_list[j*n_reps + i])
        
        reps.append(a)
    return reps


def fetch_reps(idx, reps_dir, n_reps):
    reps_file = open(f'{reps_dir}/reps_{idx}', 'rb')
    t_reps = pkl.load(reps_file)
    s_reps = pkl.load(reps_file)
    labels = pkl.load(reps_file)
    reps_file.close()

    t_reps = reshape_list(t_reps, n_reps)
    s_reps = reshape_list(s_reps, n_reps)
    return t_reps, s_reps, np.array(labels)


def compute_measures(rep_i, t_reps, s_reps, labels):
    # t_disc, _ = clustering(np.reshape(t_reps[rep_i], (len(t_reps[rep_i]),-1)), pca=True, n_components=10, n_clusters=10)
    # s_disc, _ = clustering(np.reshape(s_reps[rep_i], (len(s_reps[rep_i]),-1)), pca=True, n_components=10, n_clusters=10)
    t_disc, _ = clustering(np.reshape(t_reps[rep_i], (len(t_reps[rep_i]),-1)), pca=True, n_components=5, n_clusters=3)
    s_disc, _ = clustering(np.reshape(s_reps[rep_i], (len(s_reps[rep_i]),-1)), pca=True, n_components=5, n_clusters=3)
    print(f'Clustering complete. t_disc shape:{t_disc.shape}, s_disc shape:{s_disc.shape}')
    distrib_tsy, maps = convert_data_to_distribution(t_disc, s_disc, labels)

    print(f'distribution t,s,y shape:{distrib_tsy.shape}')
    dit_distrib_tsy = Distribution.from_ndarray(distrib_tsy)

    return get_measure(distrib_tsy)



dirs = []
save_reps_path = './aistatcam3_cifar100_2810/reps'
save_meas_path = './aistatcam3_cifar100_2810/meas100spc'
# for tag in ['tt', 'ut']:
#     for kd in ['red', 'vid', 'bas']:
#         dirs.append((f'{save_reps_path}/reps_{kd}_{tag}', f'{save_meas_path}/meas_{kd}_{tag}'))

# dirs.append((f'{save_reps_path}/reps_ted_tt', f'{save_meas_path}/meas_ted_tt'))
# dirs.append((f'{save_reps_path}/reps_ted_ut', f'{save_meas_path}/meas_ted_ut'))

dirs = [
    # (f'{save_reps_path}/reps_vid_tt_2503100037', f'{save_meas_path}/measures_vid_tt'),
    # (f'{save_reps_path}/reps_vid_ut_2503100754', f'{save_meas_path}/measures_vid_ut'),
    # (f'{save_reps_path}/reps_bas_tt_2503100335', f'{save_meas_path}/measures_bas_tt'),
]

n_reps = 3

for reps_dir, meas_dir in dirs:
    print(f'now computing: {reps_dir}')
    if os.path.exists(meas_dir) and os.path.isdir(meas_dir):
        raise FileExistsError(f"Error: The folder '{meas_dir}' already exists.")
    else:
        os.makedirs(meas_dir)
        print(f'created directory {meas_dir}')

    for rep_i in [2]:
    # for rep_i in range(3):
        for i in range(0,251,50):
        # for i in [250]:
            start_time = time.time()
            print(f'rep_i={rep_i}, i={i}')
            t_reps, s_reps, labels = fetch_reps(i, reps_dir, n_reps=n_reps)
            measures = compute_measures(rep_i, t_reps, s_reps, labels)
            
            filename = f'{meas_dir}/reps_{rep_i}_{i}'
            with open(filename, 'wb') as file:
                pkl.dump(measures, file)

            elapsed_time = time.time() - start_time
            print(f'time elapsed: {str(timedelta(seconds=elapsed_time))}')
