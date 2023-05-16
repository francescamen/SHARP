
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
import pickle
import math as mt
from os import listdir
from plots_utility import plt_doppler_antennas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('sub_dir', help='Sub directory of data')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('labels_activities', help='Files names')
    parser.add_argument('end_plt', help='End index to plot', type=int)

    args = parser.parse_args()

    feature_length = args.feature_length
    sliding = args.sliding
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8

    exp_dir = args.dir + args.sub_dir + '/'
    print(exp_dir)

    labels_activities = args.labels_activities
    csi_label_dict = []
    for lab_act in labels_activities.split(','):
        csi_label_dict.append(lab_act)
    print(csi_label_dict)

    all_files = listdir(exp_dir)
    print(all_files)

    for ilab in range(len(csi_label_dict)):
        names = []
        activity = csi_label_dict[ilab]

        start_l = 4
        end_l = start_l + len(csi_label_dict[ilab])
        for i in range(len(all_files)):
            print(all_files[i][start_l:end_l])
            if all_files[i][start_l:end_l] == csi_label_dict[ilab]:
                names.append(all_files[i][:-4])

        names.sort()

        stft_antennas = []
        for name in names:
            name_file = exp_dir + name + '.txt'
            print(name_file)
            with open(name_file, "rb") as fp:  # Pickling
                stft_sum_1 = pickle.load(fp)

            stft_sum_1[stft_sum_1 < mt.pow(10, -2.5)] = mt.pow(10, -2.5)

            stft_sum_1_log = 10*np.log10(stft_sum_1)
            middle = int(np.floor(stft_sum_1_log.shape[1] / 2))

            stft_sum_1_log = stft_sum_1_log[:min(stft_sum_1_log.shape[0], args.end_plt), :]

            stft_antennas.append(stft_sum_1_log)

        name_p = './plots/csi_doppler_activity_' + args.sub_dir + '_' + activity + '.png'
        delta_v = round(v_light / (Tc * fc * feature_length), 3)

        plt_doppler_antennas(stft_antennas, sliding, delta_v, name_p)
