
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

import numpy as np
import pickle
import tensorflow as tf


def convert_to_number(lab, csi_label_dict):
    lab_num = np.argwhere(np.asarray(csi_label_dict) == lab)[0][0]
    return lab_num


def create_windows(csi_list, labels_list, sample_length, stride_length):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[1]
        for ii in range(0, len_csi - sample_length, stride_length):
            csi_matrix_stride.append(csi_i[:, ii:ii+sample_length])
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride


def create_windows_antennas(csi_list, labels_list, sample_length, stride_length, remove_mean=False):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[2]
        for ii in range(0, len_csi - sample_length, stride_length):
            csi_wind = csi_i[:, :, ii:ii + sample_length, ...]
            if remove_mean:
                csi_mean = np.mean(csi_wind, axis=2, keepdims=True)
                csi_wind = csi_wind - csi_mean
            csi_matrix_stride.append(csi_wind)
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride


def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [item for item in labels for _ in range(num_antennas)]
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant


def load_data(csi_file_t):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    matrix_csi = tf.transpose(matrix_csi, perm=[2, 1, 0])
    matrix_csi = tf.cast(matrix_csi, tf.float32)
    return matrix_csi


def create_dataset(csi_matrix_files, labels_stride, input_shape, batch_size, shuffle, cache_file, prefetch=True,
                   repeat=True):
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride))
    py_funct = lambda csi_file, label: (tf.ensure_shape(tf.numpy_function(load_data, [csi_file], tf.float32),
                                                        input_shape), label)
    dataset_csi = dataset_csi.map(py_funct)
    dataset_csi = dataset_csi.cache(cache_file)
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=1)
    return dataset_csi


def randomize_antennas(csi_data):
    stream_order = np.random.permutation(csi_data.shape[2])
    csi_data_randomized = csi_data[:, :, stream_order]
    return csi_data_randomized


def create_dataset_randomized_antennas(csi_matrix_files, labels_stride, input_shape, batch_size, shuffle, cache_file,
                                       prefetch=True, repeat=True):
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride))
    py_funct = lambda csi_file, label: (tf.ensure_shape(tf.numpy_function(load_data, [csi_file], tf.float32),
                                                        input_shape), label)
    dataset_csi = dataset_csi.map(py_funct)
    dataset_csi = dataset_csi.cache(cache_file)

    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    if repeat:
        dataset_csi = dataset_csi.repeat()

    randomize_funct = lambda csi_data, label: (tf.ensure_shape(tf.numpy_function(randomize_antennas, [csi_data],
                                                                                 tf.float32), input_shape), label)
    dataset_csi = dataset_csi.map(randomize_funct)

    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=1)
    return dataset_csi


def load_data_single(csi_file_t, stream_a):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    matrix_csi_single = matrix_csi[stream_a, ...].T
    if len(matrix_csi_single.shape) < 3:
        matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)
    matrix_csi_single = tf.cast(matrix_csi_single, tf.float32)
    return matrix_csi_single


def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, cache_file,
                          prefetch=True, repeat=True):
    stream_ant = list(stream_ant)
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride, stream_ant))
    py_funct = lambda csi_file, label, stream: (tf.ensure_shape(tf.numpy_function(load_data_single,
                                                                                  [csi_file, stream],
                                                                                  tf.float32), input_shape), label)
    dataset_csi = dataset_csi.map(py_funct)
    dataset_csi = dataset_csi.cache(cache_file)
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=1)
    return dataset_csi
