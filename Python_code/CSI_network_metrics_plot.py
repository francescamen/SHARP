
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
from plots_utility import plt_confusion_matrix


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
    conf_matrix_max_merge = conf_matrix_dict['conf_matrix_max_merge']

    name_plot = args.name_file
    plt_confusion_matrix(activities.shape[0], conf_matrix, activities=activities, name=name_plot)

    name_plot = args.name_file + '_max_merge'
    plt_confusion_matrix(activities.shape[0], conf_matrix_max_merge, activities=activities, name=name_plot)
