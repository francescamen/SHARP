
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import math as mt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times'
rcParams['text.usetex'] = 'true'
rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
rcParams['font.size'] = 16


def plot_r_angle(complex_opt_r, start_plot, end_plot, delta_t, save_name):
    plt.figure()
    plt.stem(np.unwrap(np.angle(complex_opt_r)))
    plt.grid()
    plt.xlabel(r'delay tot [s]')
    plt.ylabel(r'$angle(\mathbf{r})|$')
    plt.xticks(np.arange(0, complex_opt_r.shape[0], 10),
               np.round(np.arange(start_plot, end_plot, 10 * delta_t), 10))
    name_file = './plots/r_angle_' + save_name + '.png'
    plt.savefig(name_file, bbox_inches='tight')


def plot_r_abs(complex_opt_r, start_plot, end_plot, delta_t, save_name):
    plt.figure()
    plt.stem(np.abs(complex_opt_r))
    plt.grid()
    plt.xlabel(r'delay tot [s]')
    plt.ylabel(r'$|\mathbf{r}|$')
    plt.xticks(np.arange(0, complex_opt_r.shape[0], 10),
               np.round(np.arange(start_plot, end_plot, 10 * delta_t), 10))
    name_file = './plots/r_abs_' + save_name + '.png'
    plt.savefig(name_file, bbox_inches='tight')


def plot_abs_comparison(H_true, H_estimated, save_name):
    plt.figure()
    plt.plot(abs(H_true), label=r'$|\mathbf{H}|$')
    plt.plot(abs(H_estimated), label=r'$|\hat{\mathbf{H}}|$')
    plt.grid()
    plt.legend()
    plt.xlabel('sub-channel')
    plt.ylabel('amplitude')
    name_file = './plots/amplitude_H_' + save_name + '.png'
    plt.savefig(name_file, bbox_inches='tight')


def plot_angle_comparison(H_true, H_estimated, save_name):
    plt.figure()
    plt.plot(np.unwrap(np.angle(H_true)), label=r'$|\mathbf{H}|$')
    plt.plot(np.unwrap(np.angle(H_estimated)), label=r'$|\hat{\mathbf{H}}|$')
    plt.grid()
    plt.legend()
    plt.xlabel('sub-channel')
    plt.ylabel('phase')
    name_file = './plots/phase_H_' + save_name + '.png'
    plt.savefig(name_file, bbox_inches='tight')


def plot_gridspec_abs(H_true, H_estimated, H_estimated_sanitized, save_name):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3, hspace=0.6, wspace=0.3, figure=fig)
    ax1 = fig.add_subplot(gs[(0, 0)])
    ax1.plot(np.abs(H_true))
    ax1.set_title('Signal with offsets')
    ax1.set_ylabel('amplitude')
    ax1.set_xlabel('sub-channel')
    ax1.grid()
    ax2 = fig.add_subplot(gs[(0, 1)])
    ax2.plot(np.abs(H_estimated))
    ax2.set_title('Signal reconstructed')
    ax2.set_ylabel('amplitude')
    ax2.set_xlabel('sub-channel')
    ax2.grid()
    ax3 = fig.add_subplot(gs[(0, 2)])
    ax3.plot(np.abs(H_estimated_sanitized))
    ax3.set_title('Signal reconstructed no offset')
    ax3.set_ylabel('amplitude')
    ax3.set_xlabel('sub-channel')
    ax3.grid()
    name_file = './plots/amplitude_comparison_' + save_name + '.png'
    plt.savefig(name_file, bbox_inches='tight')


def plot_gridspec_angle(H_true, H_estimated, H_estimated_sanitized, save_name):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3, hspace=0.6, wspace=0.3, figure=fig)
    ax1 = fig.add_subplot(gs[(0, 0)])
    ax1.plot(np.unwrap(np.angle(H_true)))
    ax1.set_title('Signal with offsets')
    ax1.set_ylabel('phase')
    ax1.set_xlabel('sub-channel')
    ax1.grid()
    ax2 = fig.add_subplot(gs[(0, 1)])
    ax2.plot(np.unwrap(np.angle(H_estimated)))
    ax2.set_title('Signal reconstructed')
    ax2.set_ylabel('phase')
    ax2.set_xlabel('sub-channel')
    ax2.grid()
    ax3 = fig.add_subplot(gs[(0, 2)])
    ax3.plot(np.unwrap(np.angle(H_estimated_sanitized)))
    ax3.set_title('Signal reconstructed no offset')
    ax3.set_ylabel('phase')
    ax3.set_xlabel('sub-channel')
    ax3.grid()
    name_file = './plots/phase_comparison_' + save_name + '.png'
    plt.savefig(name_file, bbox_inches='tight')


def plt_antennas(spectrum_list, name_plot, step=100):
    fig = plt.figure()
    gs = gridspec.GridSpec(len(spectrum_list), 1, figure=fig)
    ticks_x = np.arange(0, spectrum_list[0].shape[0], step)
    ax = []

    for p_i in range(len(spectrum_list)):
        ax1 = fig.add_subplot(gs[(p_i, 0)])
        plt1 = ax1.pcolormesh(spectrum_list[p_i].T, shading='gouraud', cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'sub-channel')
        ax1.set_xlabel(r'time [s]')
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(ticks_x * 6e-3)
        ax.append(ax1)

    for axi in ax:
        axi.label_outer()
    fig.set_size_inches(20, 10)
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_doppler_antennas(doppler_spectrum_list, sliding_lenght, delta_v, name_plot):
    if doppler_spectrum_list:
        fig = plt.figure()
        gs = gridspec.GridSpec(4, 1, figure=fig)
        step = 15
        length_v = mt.floor(doppler_spectrum_list[0].shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, doppler_spectrum_list[0].shape[0], int(doppler_spectrum_list[0].shape[0]/20))
        ax = []

        for p_i in range(len(doppler_spectrum_list)):
            ax1 = fig.add_subplot(gs[(p_i, 0)])
            plt1 = ax1.pcolormesh(doppler_spectrum_list[p_i].T, cmap='viridis', linewidth=0, rasterized=True)
            plt1.set_edgecolor('face')
            cbar1 = fig.colorbar(plt1)
            cbar1.ax.set_ylabel('power [dB]', rotation=270, labelpad=14)
            ax1.set_ylabel(r'velocity [m/s]')
            ax1.set_xlabel(r'time [s]')
            ax1.set_yticks(ticks_y + 0.5)
            ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
            ax1.set_xticks(ticks_x)
            ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))
            ax.append(ax1)

        for axi in ax:
            axi.label_outer()
        fig.set_size_inches(20, 10)
        plt.savefig(name_plot, bbox_inches='tight')
        plt.close()


def plt_confusion_matrix(number_activities, confusion_matrix, activities, name):
    confusion_matrix_normaliz_row = np.transpose(confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(-1, 1))
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(5.5, 4)
    ax = fig.add_axes((0.18, 0.15, 0.6, 0.8))
    im1 = ax.pcolor(np.linspace(0.5, number_activities + 0.5, number_activities + 1),
                    np.linspace(0.5, number_activities + 0.5, number_activities + 1),
                    confusion_matrix_normaliz_row, cmap='Blues', edgecolors='black', vmin=0, vmax=1)
    ax.set_xlabel('Actual activity', FontSize=18)
    ax.set_xticks(np.linspace(1, number_activities, number_activities))
    ax.set_xticklabels(labels=activities, FontSize=18)
    ax.set_yticks(np.linspace(1, number_activities, number_activities))
    ax.set_yticklabels(labels=activities, FontSize=18, rotation=45)
    ax.set_ylabel('Predicted activity', FontSize=18)

    for x_ax in range(confusion_matrix_normaliz_row.shape[0]):
        for y_ax in range(confusion_matrix_normaliz_row.shape[1]):
            col = 'k'
            value_c = round(confusion_matrix_normaliz_row[x_ax, y_ax], 2)
            if value_c > 0.6:
                col = 'w'
            if value_c > 0:
                ax.text(y_ax + 1, x_ax + 1, '%.2f' % value_c, horizontalalignment='center',
                        verticalalignment='center', fontsize=16, color=col)

    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.8])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.ax.set_ylabel('Accuracy', FontSize=18)
    cbar.ax.tick_params(axis="y", labelsize=16)

    plt.tight_layout()
    name_fig = './plots/cm_' + name + '.pdf'
    plt.savefig(name_fig)


def plt_doppler_activities(doppler_spectrum_list, antenna, csi_label_dict, sliding_lenght, delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(15, 3)
    widths = [1, 1, 1, 1, 1, 0.5]
    heights = [1]
    gs = fig.add_gridspec(ncols=6, nrows=1, width_ratios=widths, height_ratios=heights)
    step = 20
    step_x = 5
    ax = []
    for a_i in range(5):
        act = doppler_spectrum_list[a_i][antenna]
        length_v = mt.floor(act.shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

        ax1 = fig.add_subplot(gs[(0, a_i)])
        plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
        ax1.set_xlabel(r'time [s]')
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))

        title_p = csi_label_dict[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        if a_i == 4:
            cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.7])
            cbar1 = fig.colorbar(plt1,  cax=cbar_ax, ticks=[-12, -8, -4, 0])
            cbar1.ax.set_ylabel('normalized power [dB]')

    for axi in ax:
        axi.label_outer()
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_doppler_activities_compact(doppler_spectrum_list, antenna, csi_label_dict, sliding_lenght, delta_v,
                                       name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(8, 5.5)
    widths = [1, 1, 1, 1, 1, 1, 0.5]
    heights = [1, 1]
    gs = fig.add_gridspec(ncols=7, nrows=2, width_ratios=widths, height_ratios=heights)
    step = 20
    step_x = 5
    ax = []
    list_plts_pos_row = [0, 0, 1, 1, 1]
    list_plts_pos_col_start = [1, 3, 0, 2, 4]
    list_plts_pos_col_end = [3, 5, 2, 4, 6]
    for a_i in range(5):
        act = doppler_spectrum_list[a_i][antenna]
        length_v = mt.floor(act.shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

        ax1 = fig.add_subplot(gs[list_plts_pos_row[a_i], list_plts_pos_col_start[a_i]:list_plts_pos_col_end[a_i]])
        plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_xlabel(r'time [s]')
        ax1.set_yticks(ticks_y + 0.5)
        if a_i == 0 or a_i == 2:
            ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
            ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
        else:
            ax1.set_yticklabels([])
            ax1.set_ylabel('')
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))

        title_p = csi_label_dict[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        if a_i == 1 or a_i == 4:
            cbar_ax = fig.add_axes([0.98, 0.2, 0.02, 0.7])
            cbar1 = fig.colorbar(plt1,  cax=cbar_ax, ticks=[-12, -8, -4, 0])
            cbar1.ax.set_ylabel('normalized power [dB]')

    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_doppler_activity_single(input_a, antenna, sliding_lenght, delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(3, 3)
    step = 20
    step_x = 5
    act = input_a[antenna][:340, :]
    length_v = mt.floor(act.shape[1] / 2)
    factor_v = step * (mt.floor(length_v / step))
    ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
    ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

    ax1 = fig.add_subplot()
    plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)
    plt1.set_edgecolor('face')
    ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
    ax1.set_xlabel(r'time [s]')
    ax1.set_yticks(ticks_y + 0.5)
    ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
    ax1.set_xticks(ticks_x)
    ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 1))

    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_doppler_comparison(doppler_spectrum_list, csi_label_dict, sliding_lenght, delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(15, 3)
    widths = [1, 1, 1, 1, 1, 0.5]
    heights = [1]
    gs = fig.add_gridspec(ncols=6, nrows=1, width_ratios=widths, height_ratios=heights)
    step = 20
    step_x = 5
    ax = []
    for a_i in range(5):
        act = doppler_spectrum_list[a_i]
        length_v = mt.floor(act.shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

        ax1 = fig.add_subplot(gs[(0, int(a_i))])
        plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
        ax1.set_xlabel(r'time [s]')
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))

        title_p = csi_label_dict[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        if a_i == 4:
            cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.7])
            cbar1 = fig.colorbar(plt1,  cax=cbar_ax, ticks=[-30, -25, -20, -15, -10, -5, 0])
            cbar1.ax.set_ylabel('normalized power [dB]')

    for axi in ax:
        axi.label_outer()

    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_amplitude_phase(ampl, phase_raw, phase_proc, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(10.6, 3)
    widths = [1, 0.3, 1, 0.3, 1, 0.3]
    heights = [1]
    gs = fig.add_gridspec(ncols=6, nrows=1, width_ratios=widths,  height_ratios=heights)
    step_x = 5
    ax = []

    data_list = [ampl, phase_raw, phase_proc]
    titles = [r'amplitude', r'raw phase', r'sanitized phase']

    for p_i in range(0, 6, 2):
        ax1 = fig.add_subplot(gs[(0, p_i)])
        a_i = int(p_i/2)
        plt1 = ax1.pcolormesh(data_list[a_i], cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'sub-channel')
        ax1.set_xlabel(r'time [s]')

        ticks_y = np.asarray([0, 30, 60, 90, 122, 154, 184, 214, 244])
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(ticks_y - 122)

        ticks_x = np.arange(0, data_list[a_i].shape[1] + 1, int(data_list[a_i].shape[1]/step_x))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * 6e-3, 2))

        title_p = titles[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        axins = inset_axes(ax1,
                           width="8%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.1, 0., 1, 1),
                           bbox_transform=ax1.transAxes,
                           borderpad=0,
                           )
        cbar1 = fig.colorbar(plt1, cax=axins)
        cbar1.ax.set_ylabel('power [dB]')

    for axi in ax:
        axi.label_outer()
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_amplitude_phase_vert(ampl, phase_raw, phase_proc, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(3.1, 8)
    widths = [1]
    heights = [1, 1, 1]
    gs = fig.add_gridspec(ncols=1, nrows=3, width_ratios=widths,  height_ratios=heights)
    step_x = 5
    ax = []

    data_list = [ampl, phase_raw, phase_proc]
    titles = [r'amplitude', r'raw phase', r'sanitized phase']

    for p_i in range(0, 3):
        ax1 = fig.add_subplot(gs[(p_i, 0)])
        a_i = p_i
        plt1 = ax1.pcolormesh(data_list[a_i], cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'sub-channel')
        ax1.set_xlabel(r'time [s]')

        ticks_y = np.asarray([0, 30, 60, 90, 122, 154, 184, 214, 244])
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(ticks_y - 122)

        ticks_x = np.arange(0, data_list[a_i].shape[1] + 1, int(data_list[a_i].shape[1]/step_x))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * 6e-3, 2))

        title_p = titles[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        axins = inset_axes(ax1,
                           width="8%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.1, 0., 1, 1),
                           bbox_transform=ax1.transAxes,
                           borderpad=0,
                           )
        cbar1 = fig.colorbar(plt1, cax=axins)
        cbar1.ax.set_ylabel('power [dB]')

    for axi in ax:
        axi.label_outer()
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_amplitude_phase_horiz(ampl, phase_raw, phase_proc, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(8, 2.8)
    widths = [1, 1, 1]
    heights = [1]
    gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=widths,  height_ratios=heights)
    step_x = 5
    ax = []

    data_list = [ampl, phase_raw, phase_proc]
    titles = [r'amplitude', r'raw phase', r'sanitized phase']

    for p_i in range(0, 3):
        ax1 = fig.add_subplot(gs[(0, p_i)])
        a_i = p_i
        plt1 = ax1.pcolormesh(data_list[a_i], cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'sub-channel')
        ax1.set_xlabel(r'time [s]')

        ticks_y = np.asarray([0, 30, 60, 90, 122, 154, 184, 214, 244])
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(ticks_y - 122)

        ticks_x = np.arange(0, data_list[a_i].shape[1] + 1, int(data_list[a_i].shape[1]/step_x))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * 6e-3, 2))

        title_p = titles[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        axins = inset_axes(ax1,
                           width="100%",  # width = 5% of parent_bbox width
                           height="20%",  # height : 50%
                           bbox_to_anchor=(0.055, -0.6, 1., 0.3),
                           bbox_transform=ax1.transAxes
                           )

        cbar1 = fig.colorbar(plt1, cax=axins, orientation='horizontal')
        cbar1.ax.set_ylabel(r'[dB]')

    for axi in ax:
        axi.label_outer()
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_amplitude(ampl, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(4, 3)
    widths = [1, 0.3]
    heights = [1]
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,  height_ratios=heights)
    step_x = 5
    ax = []

    ax1 = fig.add_subplot(gs[(0, 0)])
    plt1 = ax1.pcolormesh(ampl, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
    plt1.set_edgecolor('face')
    ax1.set_ylabel(r'sub-channel')
    ax1.set_xlabel(r'time [s]')

    ticks_y = np.asarray([0, 30, 60, 90, 122, 154, 184, 214, 244])
    ax1.set_yticks(ticks_y + 0.5)
    ax1.set_yticklabels(ticks_y - 122)

    ticks_x = np.arange(0, ampl.shape[1] + 1, int(ampl.shape[1]/step_x))
    ax1.set_xticks(ticks_x)
    ax1.set_xticklabels(np.round(ticks_x * 6e-3, 2))

    ax.append(ax1)

    axins = inset_axes(ax1,
                       width="8%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.1, 0., 1, 1),
                       bbox_transform=ax1.transAxes,
                       borderpad=0,
                       )
    cbar1 = fig.colorbar(plt1, cax=axins)
    cbar1.ax.set_ylabel(r'[dB]')

    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_phase(phase_raw, phase_proc, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7, 3)
    widths = [1, 0.3, 1, 0.3]
    heights = [1]
    gs = fig.add_gridspec(ncols=4, nrows=1, width_ratios=widths,  height_ratios=heights)
    step_x = 5
    ax = []

    data_list = [phase_raw, phase_proc]
    titles = [r'raw phase', r'sanitized phase']

    for p_i in range(0, 4, 2):
        ax1 = fig.add_subplot(gs[(0, p_i)])
        a_i = int(p_i/2)
        plt1 = ax1.pcolormesh(data_list[a_i], cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'sub-channel')
        ax1.set_xlabel(r'time [s]')

        ticks_y = np.asarray([0, 30, 60, 90, 122, 154, 184, 214, 244])
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(ticks_y - 122)

        ticks_x = np.arange(0, data_list[a_i].shape[1] + 1, int(data_list[a_i].shape[1]/step_x))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * 6e-3, 2))

        title_p = titles[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        axins = inset_axes(ax1,
                           width="8%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.1, 0., 1, 1),
                           bbox_transform=ax1.transAxes,
                           borderpad=0,
                           )
        cbar1 = fig.colorbar(plt1, cax=axins)
        cbar1.ax.set_ylabel('power [dB]')

    for axi in ax:
        axi.label_outer()
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_fft_doppler_activities(doppler_spectrum_list, antenna, csi_label_dict, sliding_lenght, delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(15, 3)
    widths = [1, 1, 1, 1, 1, 0.5]
    heights = [1]
    gs = fig.add_gridspec(ncols=6, nrows=1, width_ratios=widths, height_ratios=heights)
    step = 20
    step_x = 5
    ax = []
    for a_i in range(5):
        act = doppler_spectrum_list[a_i][antenna]
        length_v = mt.floor(act.shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

        ax1 = fig.add_subplot(gs[(0, a_i)])
        plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
        ax1.set_xlabel(r'time [s]')
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))

        title_p = csi_label_dict[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        if a_i == 4:
            cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.7])
            cbar1 = fig.colorbar(plt1,  cax=cbar_ax, ticks=[-12, -8, -4, 0])
            cbar1.ax.set_ylabel('normalized power [dB]')

    for axi in ax:
        axi.label_outer()
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_fft_doppler_activities_compact(doppler_spectrum_list, antenna, csi_label_dict, sliding_lenght, delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(8, 5.5)
    widths = [1, 1, 1, 1, 1, 1, 0.5]
    heights = [1, 1]
    gs = fig.add_gridspec(ncols=7, nrows=2, width_ratios=widths, height_ratios=heights)
    step = 20
    step_x = 5
    ax = []
    list_plts_pos_row = [0, 0, 1, 1, 1]
    list_plts_pos_col_start = [1, 3, 0, 2, 4]
    list_plts_pos_col_end = [3, 5, 2, 4, 6]
    for a_i in range(5):
        act = doppler_spectrum_list[a_i][antenna]
        length_v = mt.floor(act.shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

        ax1 = fig.add_subplot(gs[list_plts_pos_row[a_i], list_plts_pos_col_start[a_i]:list_plts_pos_col_end[a_i]])
        plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
        plt1.set_edgecolor('face')
        ax1.set_xlabel(r'time [s]')
        ax1.set_yticks(ticks_y + 0.5)
        if a_i == 0 or a_i == 2:
            ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
            ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
        else:
            ax1.set_yticklabels([])
            ax1.set_ylabel('')
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))

        title_p = csi_label_dict[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        if a_i == 1 or a_i == 4:
            cbar_ax = fig.add_axes([0.98, 0.2, 0.02, 0.7])
            cbar1 = fig.colorbar(plt1,  cax=cbar_ax, ticks=[-12, -8, -4, 0])
            cbar1.ax.set_ylabel('normalized power [dB]')

    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_fft_doppler_activities_compact_2(doppler_spectrum_list, antenna, csi_label_dict, sliding_lenght,
                                         delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(6, 5.5)
    widths = [1, 1, 0.5]
    heights = [1, 1]
    gs = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths, height_ratios=heights)
    step = 20
    step_x = 5
    ax = []
    list_plts_pos_row = [0, 0, 1, 1]
    list_plts_pos_col = [0, 1, 0, 1]
    for a_i in range(4):
        act = doppler_spectrum_list[a_i][antenna]
        length_v = mt.floor(act.shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

        ax1 = fig.add_subplot(gs[list_plts_pos_row[a_i], list_plts_pos_col[a_i]])
        plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
        plt1.set_edgecolor('face')
        ax1.set_xlabel(r'time [s]')
        ax1.set_yticks(ticks_y + 0.5)
        if a_i == 0 or a_i == 2:
            ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
            ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
        else:
            ax1.set_yticklabels([])
            ax1.set_ylabel('')
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))

        title_p = csi_label_dict[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        if a_i == 1:
            cbar_ax = fig.add_axes([0.9, 0.2, 0.03, 0.7])
            cbar1 = fig.colorbar(plt1,  cax=cbar_ax, ticks=[-12, -8, -4, 0])
            cbar1.ax.set_ylabel('normalized power [dB]')

    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_fft_doppler_activity_single(input_a, antenna, sliding_lenght, delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(3, 3)
    step = 20
    step_x = 5
    act = input_a[antenna][:340, :]
    length_v = mt.floor(act.shape[1] / 2)
    factor_v = step * (mt.floor(length_v / step))
    ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
    ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

    ax1 = fig.add_subplot()
    plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
    plt1.set_edgecolor('face')
    ax1.set_ylabel(r'$v_p \cos \alpha_p$ [m/s]')
    ax1.set_xlabel(r'time [s]')
    ax1.set_yticks(ticks_y + 0.5)
    ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
    ax1.set_xticks(ticks_x)
    ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 1))

    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()
