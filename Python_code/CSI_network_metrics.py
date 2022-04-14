
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name_file', help='Name of the file')
    parser.add_argument('activities', help='Activities to be considered')
    args = parser.parse_args()

    name_file = args.name_file  # string
    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)

    folder_name = './outputs/'

    name_file = folder_name + name_file + '.txt'

    with open(name_file, "rb") as fp:  # Pickling
        conf_matrix_dict = pickle.load(fp)

    conf_matrix = conf_matrix_dict['conf_matrix']
    confusion_matrix_normaliz_row = np.transpose(conf_matrix / np.sum(conf_matrix, axis=1).reshape(-1, 1))
    accuracies = np.diag(confusion_matrix_normaliz_row)
    accuracy = conf_matrix_dict['accuracy_single']
    precision = conf_matrix_dict['precision_single']
    recall = conf_matrix_dict['recall_single']
    fscore = conf_matrix_dict['fscore_single']
    average_prec = np.mean(precision)
    average_rec = np.mean(recall)
    average_f = np.mean(recall)
    print('single antenna - average accuracy %f, average precision %f, average recall %f, average fscore %f'
          % (accuracy, average_prec, average_rec, average_f))
    print('fscores - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (fscore[0], fscore[1], fscore[2], fscore[3], fscore[4]))
    print('average fscore %f' % (np.mean(fscore)))
    print('accuracies - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (accuracies[0], accuracies[1], accuracies[2], accuracies[3], accuracies[4]))

    conf_matrix_max_merge = conf_matrix_dict['conf_matrix_max_merge']
    conf_matrix_max_merge_normaliz_row = np.transpose(conf_matrix_max_merge /
                                                      np.sum(conf_matrix_max_merge, axis=1).reshape(-1, 1))
    accuracies_max_merge = np.diag(conf_matrix_max_merge_normaliz_row)
    accuracy_max_merge = conf_matrix_dict['accuracy_max_merge']
    precision_max_merge = conf_matrix_dict['precision_max_merge']
    recall_max_merge = conf_matrix_dict['recall_max_merge']
    fscore_max_merge = conf_matrix_dict['fscore_max_merge']
    average_max_merge_prec = np.mean(precision_max_merge)
    average_max_merge_rec = np.mean(recall_max_merge)
    average_max_merge_f = np.mean(fscore_max_merge)
    print('\n-- FINAL DECISION --')
    print('max-merge - average accuracy %f, average precision %f, average recall %f, average fscore %f'
          % (accuracy_max_merge, average_max_merge_prec, average_max_merge_rec, average_max_merge_f))
    print('fscores - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (fscore_max_merge[0], fscore_max_merge[1], fscore_max_merge[2], fscore_max_merge[3], fscore_max_merge[4]))
    print('accuracies - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (accuracies_max_merge[0], accuracies_max_merge[1], accuracies_max_merge[2], accuracies_max_merge[3],
             accuracies_max_merge[4]))

    # performance assessment by changing the number of monitor antennas
    name_file = folder_name + 'change_number_antennas_' + args.name_file + '.txt'
    with open(name_file, "rb") as fp:  # Pickling
        metrics_matrix_dict = pickle.load(fp)

    average_accuracy_change_num_ant = metrics_matrix_dict['average_accuracy_change_num_ant']
    average_fscore_change_num_ant = metrics_matrix_dict['average_fscore_change_num_ant']
    print('\naccuracies - one antenna %f, two antennas %f, three antennas %f, four antennas %f'
          % (average_accuracy_change_num_ant[0], average_accuracy_change_num_ant[1], average_accuracy_change_num_ant[2],
             average_accuracy_change_num_ant[3]))
    print('fscores - one antenna %f, two antennas %f, three antennas %f, four antennas %f'
          % (average_fscore_change_num_ant[0], average_fscore_change_num_ant[1], average_fscore_change_num_ant[2],
             average_fscore_change_num_ant[3]))
