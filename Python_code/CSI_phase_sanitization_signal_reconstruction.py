
"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import scipy.io as sio
from os import listdir, path
import pickle
import math as mt
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('dir_save', help='Directory to save processed data')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('start_idx', help='Start index', type=int)
    parser.add_argument('end_idx', help='End index from the end', type=int)
    args = parser.parse_args()

    exp_dir = args.dir
    save_dir = args.dir_save
    names = []

    all_files = listdir(exp_dir)
    for i in range(len(all_files)):
        if all_files[i].startswith('Tr') and all_files[i].endswith('.txt'):
            names.append(all_files[i][:-4])

    for name in names:
        name_f = name[10:] + '.mat'
        stop = False
        sub_dir_name = name_f[0:3]
        subdir_path = save_dir + sub_dir_name

        complete_path = subdir_path + '/' + name_f
        print(complete_path)
        if path.isfile(complete_path):
            stop = True

        if stop:
            print('Already processed')
            continue

        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

        name_file_save = subdir_path + '/' + name_f
        name_file = exp_dir + name + '.txt'

        with open(name_file, "rb") as fp:  # Unpickling
            H_est = pickle.load(fp)

        end_H = H_est.shape[1]
        H_est = H_est[:, args.start_idx:end_H-args.end_idx]
        F_frequency = 256
        csi_matrix_processed = np.zeros((H_est.shape[1], F_frequency, 2))

        # AMPLITUDE
        csi_matrix_processed[:, 6:-5, 0] = np.abs(H_est[6:-5, :]).T

        # PHASE
        phase_before = np.unwrap(np.angle(H_est[6:-5, :]), axis=0)
        phase_err_tot = np.diff(phase_before, axis=1)
        ones_vector = np.ones((2, phase_before.shape[0]))
        ones_vector[1, :] = np.arange(0, phase_before.shape[0])
        for tidx in range(1, phase_before.shape[1]):
            stop = False
            idx_prec = -1
            while not stop:
                phase_err = phase_before[:, tidx] - phase_before[:, tidx - 1]
                diff_phase_err = np.diff(phase_err)
                idxs_invert_up = np.argwhere(diff_phase_err > 0.9 * mt.pi)[:, 0]
                idxs_invert_down = np.argwhere(diff_phase_err < -0.9 * mt.pi)[:, 0]
                if idxs_invert_up.shape[0] > 0:
                    idx_act = idxs_invert_up[0]
                    if idx_act == idx_prec:  # to avoid a continuous jump
                        stop = True
                    else:
                        phase_before[idx_act + 1:, tidx] = phase_before[idx_act + 1:, tidx] \
                                                           - 2 * mt.pi
                        idx_prec = idx_act
                elif idxs_invert_down.shape[0] > 0:
                    idx_act = idxs_invert_down[0]
                    if idx_act == idx_prec:
                        stop = True
                    else:
                        phase_before[idx_act + 1:, tidx] = phase_before[idx_act + 1:, tidx] \
                                                           + 2 * mt.pi
                        idx_prec = idx_act
                else:
                    stop = True
        for tidx in range(1, H_est.shape[1] - 1):
            val_prec = phase_before[:, tidx - 1:tidx]
            val_act = phase_before[:, tidx:tidx + 1]
            error = val_act - val_prec
            temp2 = np.linalg.lstsq(ones_vector.T, error)[0]
            phase_before[:, tidx] = phase_before[:, tidx] - (np.dot(ones_vector.T, temp2)).T

        csi_matrix_processed[:, 6:-5, 1] = phase_before.T

        mdic = {"csi_matrix_processed": csi_matrix_processed[:, 6:-5, :]}
        sio.savemat(name_file_save, mdic)
