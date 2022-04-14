
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
from os import listdir
import pickle
from os import path


def hampel_filter(input_matrix, window_size, n_sigmas=3):
    n = input_matrix.shape[1]
    new_matrix = np.zeros_like(input_matrix)
    k = 1.4826  # scale factor for Gaussian distribution

    for ti in range(n):
        start_time = max(0, ti - window_size)
        end_time = min(n, ti + window_size)
        x0 = np.nanmedian(input_matrix[:, start_time:end_time], axis=1, keepdims=True)
        s0 = k * np.nanmedian(np.abs(input_matrix[:, start_time:end_time] - x0), axis=1)
        mask = (np.abs(input_matrix[:, ti] - x0[:, 0]) > n_sigmas * s0)
        new_matrix[:, ti] = mask*x0[:, 0] + (1 - mask)*input_matrix[:, ti]

    return new_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('all_dir', help='All the files in the directory, default no', type=int, default=0)
    parser.add_argument('name', help='Name of experiment file')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('start_idx', help='Idx where start processing for each stream', type=int)
    args = parser.parse_args()

    exp_dir = args.dir
    names = []

    if args.all_dir:
        all_files = listdir(exp_dir)
        mat_files = []
        for i in range(len(all_files)):
            if all_files[i].endswith('.mat'):
                names.append(all_files[i][:-4])
    else:
        names.append(args.name)

    for name in names:
        name_file = './phase_processing/signal_' + name + '.txt'
        if path.exists(name_file):
            print('Already processed')
            continue

        csi_buff_file = exp_dir + name + ".mat"
        csi_buff = sio.loadmat(csi_buff_file)
        csi_buff = (csi_buff['csi_buff'])
        csi_buff = np.fft.fftshift(csi_buff, axes=1)

        delete_idxs = np.argwhere(np.sum(csi_buff, axis=1) == 0)[:, 0]
        csi_buff = np.delete(csi_buff, delete_idxs, axis=0)

        delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)

        n_ss = args.nss
        n_core = args.ncore
        n_tot = n_ss * n_core

        start = args.start_idx  # 1000
        end = int(np.floor(csi_buff.shape[0]/n_tot))
        signal_complete = np.zeros((csi_buff.shape[1] - delete_idxs.shape[0], end-start, n_tot), dtype=complex)

        for stream in range(0, n_tot):
            signal_stream = csi_buff[stream:end*n_tot + 1:n_tot, :][start:end, :]
            signal_stream[:, 64:] = - signal_stream[:, 64:]

            signal_stream = np.delete(signal_stream, delete_idxs, axis=1)
            mean_signal = np.mean(np.abs(signal_stream), axis=1, keepdims=True)
            H_m = signal_stream/mean_signal

            signal_complete[:, :, stream] = H_m.T

        name_file = './phase_processing/signal_' + name + '.txt'
        with open(name_file, "wb") as fp:  # Pickling
            pickle.dump(signal_complete, fp)
