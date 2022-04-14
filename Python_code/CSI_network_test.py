
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
from sklearn.metrics import confusion_matrix
import os
from dataset_utility import create_dataset_single, expand_antennas
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Subdirs for testing')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int)
    parser.add_argument('channels', help='Number of channels', type=int)
    parser.add_argument('batch_size', help='Number of samples in a batch', type=int)
    parser.add_argument('num_tot', help='Number of antenna * number of spatial streams', type=int)
    parser.add_argument('name_base', help='Name base for the files')
    parser.add_argument('activities', help='Activities to be considered')
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)

    suffix = '.txt'

    name_base = args.name_base
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_complete.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) + '_cache_complete.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_complete.index')

    subdirs_complete = args.subdirs  # string
    labels_complete = []
    all_files_complete = []
    sample_length = args.sample_length
    feature_length = args.feature_length
    channels = args.channels
    num_antennas = args.num_tot
    input_shape = (num_antennas, sample_length, feature_length, channels)
    input_network = (sample_length, feature_length, channels)
    batch_size = args.batch_size
    output_shape = activities.shape[0]
    labels_considered = np.arange(output_shape)
    activities = activities[labels_considered]

    for sdir in subdirs_complete.split(','):
        exp_save_dir = args.dir + sdir + '/'
        dir_complete = args.dir + sdir + '/complete_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_complete_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_complete.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_complete_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_complete.extend(pickle.load(fp))

    file_complete_selected = [all_files_complete[idx] for idx in range(len(labels_complete)) if labels_complete[idx] in
                              labels_considered]
    labels_complete_selected = [labels_complete[idx] for idx in range(len(labels_complete)) if labels_complete[idx] in
                                labels_considered]

    file_complete_selected_expanded, labels_complete_selected_expanded, stream_ant_complete = \
        expand_antennas(file_complete_selected, labels_complete_selected, num_antennas)

    dataset_csi_complete = create_dataset_single(file_complete_selected_expanded, labels_complete_selected_expanded,
                                                 stream_ant_complete, input_network, batch_size, shuffle=False,
                                                 cache_file=name_base + '_' + str(csi_act) + '_cache_complete')

    name_model = name_base + '_' + str(csi_act) + '_network.h5'
    csi_model = load_model(name_model)

    num_samples_complete = len(file_complete_selected_expanded)
    lab_complete, count_complete = np.unique(labels_complete_selected_expanded, return_counts=True)
    complete_steps_per_epoch = int(np.ceil(num_samples_complete / batch_size))

    # complete
    complete_labels_true = np.array(labels_complete_selected_expanded)
    complete_prediction_list = csi_model.predict(dataset_csi_complete,
                                                 steps=complete_steps_per_epoch)[:complete_labels_true.shape[0]]

    complete_labels_pred = np.argmax(complete_prediction_list, axis=1)

    conf_matrix = confusion_matrix(complete_labels_true, complete_labels_pred, labels=labels_considered)
    precision, recall, fscore, _ = precision_recall_fscore_support(complete_labels_true,
                                                                   complete_labels_pred,
                                                                   labels=labels_considered)
    accuracy = accuracy_score(complete_labels_true, complete_labels_pred)

    # merge antennas
    labels_true_merge = np.array(labels_complete_selected)
    pred_max_merge = np.zeros_like(labels_complete_selected)
    for i_lab in range(len(labels_complete_selected)):
        pred_antennas = complete_prediction_list[i_lab*num_antennas:(i_lab+1)*num_antennas, :]
        sum_pred = np.sum(pred_antennas, axis=0)
        lab_merge_max = np.argmax(sum_pred)

        pred_max_antennas = complete_labels_pred[i_lab*num_antennas:(i_lab+1)*num_antennas]
        lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
        lab_max_merge = -1
        if lab_unique.shape[0] > 1:
            count_argsort = np.flip(np.argsort(count))
            count_sort = count[count_argsort]
            lab_unique_sort = lab_unique[count_argsort]
            if count_sort[0] == count_sort[1] or lab_unique.shape[0] > 2:  # ex aequo between two labels
                lab_max_merge = lab_merge_max
            else:
                lab_max_merge = lab_unique_sort[0]
        else:
            lab_max_merge = lab_unique[0]
        pred_max_merge[i_lab] = lab_max_merge

    conf_matrix_max_merge = confusion_matrix(labels_true_merge, pred_max_merge, labels_considered)
    precision_max_merge, recall_max_merge, fscore_max_merge, _ = \
        precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=labels_considered)
    accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

    metrics_matrix_dict = {'conf_matrix': conf_matrix,
                           'accuracy_single': accuracy,
                           'precision_single': precision,
                           'recall_single': recall,
                           'fscore_single': fscore,
                           'conf_matrix_max_merge': conf_matrix_max_merge,
                           'accuracy_max_merge': accuracy_max_merge,
                           'precision_max_merge': precision_max_merge,
                           'recall_max_merge': recall_max_merge,
                           'fscore_max_merge': fscore_max_merge}

    name_file = './outputs/complete_different_' + str(csi_act) + '_' + subdirs_complete + '_band_' + str(bandwidth) \
                + '_subband_' + str(sub_band) + suffix
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)
    print('accuracy', accuracy_max_merge)
    print('fscore', fscore_max_merge)
    print(conf_matrix_max_merge)

    # impact of the number of antennas
    one_antenna = [[0], [1], [2], [3]]
    two_antennas = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    three_antennas = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    four_antennas = [[0, 1, 2, 3]]
    seq_ant_list = [one_antenna, two_antennas, three_antennas, four_antennas]
    average_accuracy_change_num_ant = np.zeros((num_antennas, ))
    average_fscore_change_num_ant = np.zeros((num_antennas, ))
    labels_true_merge = np.array(labels_complete_selected)
    for ant_n in range(num_antennas):
        seq_ant = seq_ant_list[ant_n]
        num_seq = len(seq_ant)
        for seq_n in range(num_seq):
            pred_max_merge = np.zeros((len(labels_complete_selected), ))
            ants_selected = seq_ant[seq_n]
            for i_lab in range(len(labels_complete_selected)):
                pred_antennas = complete_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
                pred_antennas = pred_antennas[ants_selected, :]

                lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

                pred_max_antennas = complete_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
                pred_max_antennas = pred_max_antennas[ants_selected]
                lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
                lab_max_merge = -1
                if lab_unique.shape[0] > 1:
                    count_argsort = np.flip(np.argsort(count))
                    count_sort = count[count_argsort]
                    lab_unique_sort = lab_unique[count_argsort]
                    if count_sort[0] == count_sort[1] or lab_unique.shape[0] > ant_n - 1:  # ex aequo between two labels
                        lab_max_merge = lab_merge_max
                    else:
                        lab_max_merge = lab_unique_sort[0]
                else:
                    lab_max_merge = lab_unique[0]
                pred_max_merge[i_lab] = lab_max_merge

            _, _, fscore_max_merge, _ = precision_recall_fscore_support(labels_true_merge, pred_max_merge,
                                                                        labels=[0, 1, 2, 3, 4])
            accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

            average_accuracy_change_num_ant[ant_n] += accuracy_max_merge
            average_fscore_change_num_ant[ant_n] += np.mean(fscore_max_merge)

        average_accuracy_change_num_ant[ant_n] = average_accuracy_change_num_ant[ant_n] / num_seq
        average_fscore_change_num_ant[ant_n] = average_fscore_change_num_ant[ant_n] / num_seq

    metrics_matrix_dict = {'average_accuracy_change_num_ant': average_accuracy_change_num_ant,
                           'average_fscore_change_num_ant': average_fscore_change_num_ant}

    name_file = './outputs/change_number_antennas_complete_different_' + str(csi_act) + '_' + subdirs_complete + \
                '_band_' + str(bandwidth) + '_subband_' + str(sub_band) + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)
