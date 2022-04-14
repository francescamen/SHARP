
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
import math as mt
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.signal.windows import hann
import pickle
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Sub-directories')
    parser.add_argument('dir_doppler', help='Directory to save the Doppler data')
    parser.add_argument('start', help='Start processing', type=int)
    parser.add_argument('end', help='End processing (samples from the end)', type=int)
    parser.add_argument('sample_length', help='Number of packet in a sample', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('noise_level', help='Level for the noise to be removed', type=float)
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)
    args = parser.parse_args()

    num_symbols = args.sample_length  # 51
    middle = int(mt.floor(num_symbols / 2))

    Tc = 6e-3
    fc = 5e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * num_symbols), 3)

    sliding = args.sliding
    noise_lev = args.noise_level
    bandwidth = args.bandwidth
    sub_band = args.sub_band

    list_subdir = args.subdirs

    for subdir in list_subdir.split(','):
        path_doppler = args.dir_doppler + subdir
        if not os.path.exists(path_doppler):
            os.mkdir(path_doppler)

        exp_dir = args.dir + subdir + '/'

        names = []
        all_files = os.listdir(exp_dir)
        for i in range(len(all_files)):
            names.append(all_files[i][:-4])

        for name in names:
            path_doppler_name = path_doppler + '/' + name + '.txt'
            if os.path.exists(path_doppler_name):
                continue

            print(path_doppler_name)
            name_file = exp_dir + name + '.mat'
            mdic = sio.loadmat(name_file)
            csi_matrix_processed = mdic['csi_matrix_processed']

            csi_matrix_processed = csi_matrix_processed[args.start:-args.end, :, :]

            csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(csi_matrix_processed[:, :, 0],
                                                                                    axis=1,  keepdims=True)

            csi_matrix_complete = csi_matrix_processed[:, :, 0]*np.exp(1j*csi_matrix_processed[:, :, 1])

            if bandwidth == 40:
                if sub_band == 1:
                    selected_subcarriers_idxs = np.arange(0, 117, 1)
                elif sub_band == 2:
                    selected_subcarriers_idxs = np.arange(128, 245, 1)
                num_selected_subcarriers = selected_subcarriers_idxs.shape[0]
                csi_matrix_complete = csi_matrix_complete[:, selected_subcarriers_idxs]
            elif bandwidth == 20:
                if sub_band == 1:
                    selected_subcarriers_idxs = np.arange(0, 57, 1)
                elif sub_band == 2:
                    selected_subcarriers_idxs = np.arange(60, 117, 1)
                elif sub_band == 3:
                    selected_subcarriers_idxs = np.arange(128, 185, 1)
                elif sub_band == 4:
                    selected_subcarriers_idxs = np.arange(188, 245, 1)
                num_selected_subcarriers = selected_subcarriers_idxs.shape[0]
                csi_matrix_complete = csi_matrix_complete[:, selected_subcarriers_idxs]

            csi_d_profile_list = []
            for i in range(0, csi_matrix_complete.shape[0]-num_symbols, sliding):
                csi_matrix_cut = csi_matrix_complete[i:i+num_symbols, :]
                csi_matrix_cut = np.nan_to_num(csi_matrix_cut)

                hann_window = np.expand_dims(hann(num_symbols), axis=-1)
                csi_matrix_wind = np.multiply(csi_matrix_cut, hann_window)
                csi_doppler_prof = fft(csi_matrix_wind, n=100, axis=0)
                csi_doppler_prof = fftshift(csi_doppler_prof, axes=0)

                csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof))
                csi_d_map = np.sum(csi_d_map, axis=1)
                csi_d_profile_list.append(csi_d_map)
            csi_d_profile_array = np.asarray(csi_d_profile_list)
            csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
            csi_d_profile_array = csi_d_profile_array/csi_d_profile_array_max
            csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)

            with open(path_doppler_name, "wb") as fp:  # Pickling
                pickle.dump(csi_d_profile_array, fp)
