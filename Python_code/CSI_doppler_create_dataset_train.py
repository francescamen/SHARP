
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
import glob
import os
import numpy as np
import pickle
import math as mt
import shutil
from dataset_utility import create_windows_antennas, convert_to_number


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Sub-directories')
    parser.add_argument('sample_lengths', help='Number of packets in a sample', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('windows_length', help='Number of samples per window', type=int)
    parser.add_argument('stride_lengths', help='Number of samples to stride', type=int)
    parser.add_argument('labels_activities', help='Labels of the activities to be considered')
    parser.add_argument('n_tot', help='Number of streams * number of antennas', type=int)
    args = parser.parse_args()

    labels_activities = args.labels_activities
    csi_label_dict = []
    for lab_act in labels_activities.split(','):
        csi_label_dict.append(lab_act)

    activities = np.asarray(labels_activities)

    n_tot = args.n_tot
    num_packets = args.sample_lengths  # 51
    middle = int(np.floor(num_packets / 2))
    list_subdir = args.subdirs  # string

    for subdir in list_subdir.split(','):
        exp_dir = args.dir + subdir + '/'

        path_train = exp_dir + 'train_antennas_' + str(activities)
        path_val = exp_dir + 'val_antennas_' + str(activities)
        path_test = exp_dir + 'test_antennas_' + str(activities)
        paths = [path_train, path_val, path_test]
        for pat in paths:
            if os.path.exists(pat):
                remove_files = glob.glob(pat + '/*')
                for f in remove_files:
                    os.remove(f)
            else:
                os.mkdir(pat)

        path_complete = exp_dir + 'complete_antennas_' + str(activities)
        if os.path.exists(path_complete):
            shutil.rmtree(path_complete)

        names = []
        all_files = os.listdir(exp_dir)
        for i in range(len(all_files)):
            if all_files[i].startswith('S'):
                names.append(all_files[i][:-4])
        names.sort()

        csi_matrices = []
        labels = []
        lengths = []
        label = 'null'
        prev_label = label
        csi_matrix = []
        processed = False
        for i_name, name in enumerate(names):
            if i_name % n_tot == 0 and i_name != 0 and processed:
                ll = csi_matrix[0].shape[1]

                for i_ant in range(1, n_tot):
                    if ll != csi_matrix[i_ant].shape[1]:
                        break
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)
                csi_matrix = []

            label = name[4]

            if label not in csi_label_dict:
                processed = False
                continue
            processed = True

            print(name)

            label = convert_to_number(label, csi_label_dict)
            if i_name % n_tot == 0:
                prev_label = label
            elif label != prev_label:
                print('error in ' + str(name))
                break

            name_file = exp_dir + name + '.txt'
            with open(name_file, "rb") as fp:  # Unpickling
                stft_sum_1 = pickle.load(fp)

            stft_sum_1_mean = stft_sum_1 - np.mean(stft_sum_1, axis=0, keepdims=True)

            csi_matrix.append(stft_sum_1_mean.T)

        error = False
        if processed:
            # for the last block
            if len(csi_matrix) < n_tot:
                print('error in ' + str(name))
            ll = csi_matrix[0].shape[1]

            for i_ant in range(1, n_tot):
                if ll != csi_matrix[i_ant].shape[1]:
                    print('error in ' + str(name))
                    error = True
            if not error:
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)

        if not error:
            lengths = np.asarray(lengths)
            length_min = np.min(lengths)

            csi_train = []
            csi_val = []
            csi_test = []
            length_train = []
            length_val = []
            length_test = []
            for i in range(len(labels)):
                ll = lengths[i]
                train_len = int(np.floor(ll * 0.6))
                length_train.append(train_len)
                csi_train.append(csi_matrices[i][:, :, :train_len])

                start_val = train_len + mt.ceil(num_packets/args.sliding)
                val_len = int(np.floor(ll * 0.2))
                length_val.append(val_len)
                csi_val.append(csi_matrices[i][:, :, start_val:start_val + val_len])

                start_test = start_val + val_len + mt.ceil(num_packets/args.sliding)
                length_test.append(ll - val_len - train_len - 2*mt.ceil(num_packets/args.sliding))
                csi_test.append(csi_matrices[i][:, :, start_test:])

            window_length = args.windows_length  # number of windows considered
            stride_length = args.stride_lengths

            list_sets_name = ['train', 'val', 'test']
            list_sets = [csi_train, csi_val, csi_test]
            list_sets_lengths = [length_train, length_val, length_test]

            for set_idx in range(3):
                csi_matrices_set, labels_set = create_windows_antennas(list_sets[set_idx], labels, window_length,
                                                                       stride_length, remove_mean=False)

                num_windows = np.floor((np.asarray(list_sets_lengths[set_idx]) - window_length) / stride_length + 1)
                if not len(csi_matrices_set) == np.sum(num_windows):
                    print('ERROR - shapes mismatch')

                names_set = []
                suffix = '.txt'
                for ii in range(len(csi_matrices_set)):
                    name_file = exp_dir + list_sets_name[set_idx] + '_antennas_' + str(activities) + '/' + \
                                str(ii) + suffix
                    names_set.append(name_file)
                    with open(name_file, "wb") as fp:  # Pickling
                        pickle.dump(csi_matrices_set[ii], fp)
                name_labels = exp_dir + '/labels_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
                with open(name_labels, "wb") as fp:  # Pickling
                    pickle.dump(labels_set, fp)
                name_f = exp_dir + '/files_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(names_set, fp)
                name_f = exp_dir + '/num_windows_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(num_windows, fp)
