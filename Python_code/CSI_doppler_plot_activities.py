
import argparse
import numpy as np
import pickle
from os import listdir
from plots_utility import plt_fft_doppler_activities, plt_fft_doppler_activities_compact, \
    plt_fft_doppler_activity_single, plt_fft_doppler_activities_compact_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('sub_dir', help='Sub directory of data')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('labels_activities', help='Files names')
    parser.add_argument('start_plt', help='Start index to plot', type=int)
    parser.add_argument('end_plt', help='End index to plot', type=int)

    args = parser.parse_args()

    activities = np.asarray(['empty', 'sitting', 'walking', 'running', 'jumping'])

    feature_length = args.feature_length
    sliding = args.sliding
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8

    exp_dir = args.dir + args.sub_dir + '/'

    labels_activities = args.labels_activities
    csi_label_dict = []
    for lab_act in labels_activities.split(','):
        csi_label_dict.append(lab_act)

    traces_activities = []
    for ilab in range(len(csi_label_dict)):
        names = []
        all_files = listdir(exp_dir)
        activity = csi_label_dict[ilab]

        start_l = 4
        end_l = start_l + len(csi_label_dict[ilab])
        for i in range(len(all_files)):
            if all_files[i][start_l:end_l] == csi_label_dict[ilab] and all_files[i][-5] != 'p':
                names.append(all_files[i][:-4])

        names.sort()

        stft_antennas = []
        for name in names:
            name_file = exp_dir + name + '.txt'

            with open(name_file, "rb") as fp:  # Pickling
                stft_sum_1 = pickle.load(fp)

            stft_sum_1_log = 10*np.log10(stft_sum_1)

            stft_sum_1_log = stft_sum_1_log[args.start_plt:min(stft_sum_1_log.shape[0], args.end_plt), :]

            stft_antennas.append(stft_sum_1_log)

        traces_activities.append(stft_antennas)

    name_p = './plots/csi_doppler_activities_' + args.sub_dir + '.pdf'
    delta_v = round(v_light / (Tc * fc * feature_length), 3)
    antenna = 0
    plt_fft_doppler_activities(traces_activities, antenna, activities, sliding, delta_v, name_p)

    name_p = './plots/csi_doppler_activities_' + '_' + args.sub_dir + '_compact.pdf'
    plt_fft_doppler_activities_compact(traces_activities, antenna, activities, sliding, delta_v, name_p)

    traces_activities_reduced = traces_activities
    del traces_activities_reduced[2]
    name_p = './plots/csi_doppler_activities_' + '_' + args.sub_dir + '_compact_2.pdf'
    plt_fft_doppler_activities_compact_2(traces_activities_reduced, antenna,
                                         np.asarray(['empty', 'sitting', 'running', 'jumping']),
                                         sliding, delta_v, name_p)

    name_p = './plots/csi_doppler_single_act.pdf'
    antenna = 1
    plt_fft_doppler_activity_single(traces_activities[4], antenna, sliding, delta_v, name_p)
