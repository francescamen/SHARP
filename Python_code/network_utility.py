
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

import tensorflow as tf


def conv2d_bn(x_in, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', bn=False, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(x_in)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x


def reduction_a_block_small(x_in, base_name):
    x1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(x_in)

    x2 = conv2d_bn(x_in, 5, (2, 2), strides=(2, 2), padding='valid', name=base_name + 'conv2_1_res_a')

    x3 = conv2d_bn(x_in, 3, (1, 1), name=base_name + 'conv3_1_res_a')
    x3 = conv2d_bn(x3, 6, (2, 2), name=base_name + 'conv3_2_res_a')
    x3 = conv2d_bn(x3, 9, (4, 4), strides=(2, 2), padding='same', name=base_name + 'conv3_3_res_a')

    x4 = tf.keras.layers.Concatenate()([x1, x2, x3])
    return x4


def csi_network_inc_res(input_sh, output_sh):
    x_input = tf.keras.Input(input_sh)

    x2 = reduction_a_block_small(x_input, base_name='1st')

    x3 = conv2d_bn(x2, 3, (1, 1), name='conv4')

    x = tf.keras.layers.Flatten()(x3)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None, name='dense2')(x)
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_model')
    return model
