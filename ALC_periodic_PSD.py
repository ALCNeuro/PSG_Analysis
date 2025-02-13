#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:44:41 2024

@author: arthurlecoz

ALC_periodic_PSD.py

"""
# %% Paths
import mne, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.stats import sem
from fooof import FOOOF
from fooof.bands import Bands
import pickle
from fooof.sim.gen import gen_aperiodic
from statsmodels.nonparametric.smoothers_lowess import lowess

from highlight_text import ax_text, fig_text
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager 
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')
prop = font_manager.FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()


root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = f"{root_dir}/Raw"
preproc_dir = f"{root_dir}/Preproc/"
fig_dir = f"{root_dir}/Figs/"

hi_cns_files = glob(os.path.join(
    preproc_dir, "hi_cns", "*PSG1*.fif")
    )
n1_files = glob(os.path.join(
    preproc_dir, "*N*PSG1*.fif")
    )

epochs_files = hi_cns_files+n1_files

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

subtypes = ["C1", "HI", "N1"]
channels = ["F3", "C3", "O1"]
# psd_palette = ["#5e6472", "#faa307"]
# sem_palette = ['#dee0e2', '#FEECCD']
psd_palette = ["#5e6472", "#faa307"]
sem_palette = ['#dee0e2', '#FEECCD']

freqs = np.linspace(0.5, 40, 159)

bands = Bands({
    "delta" : (.5, 4),
    "theta" : (4, 8),
    "alpha" : (8, 12),
    "sigma" : (12, 16),
    "iota" : (25, 35)
    })

method = "welch"
fmin = 0.5
fmax = 40
n_fft = 1024
n_per_seg = n_fft
n_overlap = int(n_per_seg/2)
window = "hamming"

# %% Loop

big_dic_psd_savepath = os.path.join(
    fig_dir, "fooof_psd_flat_spectra_2.pickle"
    )

stages = ["WAKE", "N1", "N2", "N3", "REM"]

# big_dic = {subtype : {stage : {chan : [] for chan in channels}
#                       for stage in stages} for subtype in subtypes}

    
def compute_periodic_psd(file) :
    
    # if subtype.startswith('N') : continue
    sub_id = file.split('/')[-1].split('_PSG1')[0]
    
    this_subject_savepath = os.path.join(
        fig_dir, "fooof", f"{sub_id}_periodic_psd.pickle"
        )
    
    temp_dic = {stage : {chan : [] for chan in channels}
                          for stage in stages}
    
    thisDemo = df_demographics.loc[df_demographics.code == sub_id]
    # age = thisDemo.age.iloc[0]
    # genre = thisDemo.sexe.iloc[0]
    print(f"...processing {sub_id}")
    
    epochs = mne.read_epochs(file, preload = True)
    epochs = epochs[epochs.metadata.divBin2 != "noBin"]
    # sf = epochs.info['sfreq']
    
    # metadata = epochs.metadata.reset_index()
        
    if len(stages) < 5 :
        print("not adapted yet...")
        # for stage in ["WAKE", "N1", "REM"] :
        #     if stage not in epochs.metadata.scoring.unique() : continue
        #     else :
        #         temp_list = []
        #         temp_power = epochs[stage].compute_psd(
        #                 method = method,
        #                 fmin = fmin, 
        #                 fmax = fmax,
        #                 n_fft = n_fft,
        #                 n_overlap = n_overlap,
        #                 n_per_seg = n_per_seg,
        #                 window = window,
        #                 picks = channels
        #                 )
        #         for i_epoch in range(len(epochs[stage])) :
        #             this_power = temp_power[i_epoch]
        #             for i_ch, channel in enumerate(channels) :
        #                 psd = lowess(np.squeeze(this_power.copy().pick(channel).get_data()), freqs, 0.075)[:, 1]
        #                 fm = FOOOF(peak_width_limits = [1, 15], aperiodic_mode="fixed")
        #                 fm.add_data(freqs, psd)
        #                 fm.fit()
                        
        #                 init_ap_fit = gen_aperiodic(
        #                     fm.freqs, 
        #                     fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
        #                     )
        #                 # Recompute the flattened spectrum using the initial aperiodic fit
        #                 init_flat_spec = fm.power_spectrum - init_ap_fit
        #                 temp_list.append(init_flat_spec)
        #             big_dic[subtype][stage][channel].append(np.nanmean(temp_list, axis = 0))
    
        # temp_list = []
        # temp_power = epochs["N2", "N3"].compute_psd(
        #             method = method,
        #             fmin = fmin, 
        #             fmax = fmax,
        #             n_fft = n_fft,
        #             n_overlap = n_overlap,
        #             n_per_seg = n_per_seg,
        #             window = window,
        #             picks = channels
        #             )
        # for i_epoch in range(len(epochs["N2", "N3"])) :
        #     this_power = temp_power[i_epoch]
        #     for i_ch, channel in enumerate(channels) :
        #         psd = lowess(np.squeeze(
        #             this_power.copy().pick(channel).get_data()), 
        #             freqs, 0.075)[:, 1]
        #         fm = FOOOF(peak_width_limits = [.5, 15], aperiodic_mode="fixed")
        #         fm.add_data(freqs, psd)
        #         fm.fit()
                
        #         init_ap_fit = gen_aperiodic(
        #             fm.freqs, 
        #             fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
        #             )
        #         # Recompute the flattened spectrum using the initial aperiodic fit
        #         init_flat_spec = fm.power_spectrum - init_ap_fit
        #         temp_list.append(init_flat_spec)
                
        #     big_dic[subtype]["N2_N3"][channel].append(np.nanmean(temp_list, axis = 0))
    else : 
        for stage in stages :
            print(f'processing stage {stage}')
            if stage not in epochs.metadata.scoring.unique() : continue
            else : 
                temp_list = []
                temp_power = epochs[stage].compute_psd(
                        method = method,
                        fmin = fmin, 
                        fmax = fmax,
                        n_fft = n_fft,
                        n_overlap = n_overlap,
                        n_per_seg = n_per_seg,
                        window = window,
                        picks = channels
                        )
                for i_ch, channel in enumerate(channels) :
                    print(f'processing stage {channel}')
                    for i_epoch in range(len(epochs[stage])) :
                        this_power = temp_power[i_epoch]                        
                        
                        psd = lowess(np.squeeze(
                            this_power.copy().pick(channel).get_data()), 
                            freqs, 0.075)[:, 1]
                        
                        if np.any(psd < 0) :
                            for id_0 in np.where(psd<0)[0] :
                                psd[id_0] = abs(psd).min()
                                
                        fm = FOOOF(peak_width_limits = [.5, 15], aperiodic_mode="fixed")
                        fm.add_data(freqs, psd)
                        fm.fit()
                        
                        init_ap_fit = gen_aperiodic(
                            fm.freqs, 
                            fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
                            )
                        
                        init_flat_spec = fm.power_spectrum - init_ap_fit
                        temp_list.append(init_flat_spec)
                    temp_dic[stage][channel].append(
                        np.nanmean(temp_list, axis = 0)
                        )
    with open (this_subject_savepath, 'wb') as handle:
        pickle.dump(temp_dic, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    from glob import glob
    # Get the list of EEG files
    eeg_files = epochs_files
    
    # Set up a pool of worker processes
    pool = multiprocessing.Pool(processes = 4)
    
    # Process the EEG files in parallel
    pool.map(compute_periodic_psd, eeg_files)
    
    # Clean up the pool of worker processes
    pool.close()
    pool.join()
    
# %% 

big_dic = ''
                
with open(big_dic_psd_savepath, 'wb') as handle:
    pickle.dump(big_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# %% 

big_av_psd_savepath = os.path.join(
    fig_dir, "fooof_averaged_psd_flat_spectra_2.pickle"
    )
big_av_sem_savepath = os.path.join(
    fig_dir, "fooof_sem_flat_spectra_2.pickle"
    )

dic_psd = {subtype : {stage : {chan : [] for chan in channels}
                      for stage in stages} for subtype in subtypes}
dic_sem = {subtype : {stage : {chan : [] for chan in channels}
                      for stage in stages} for subtype in subtypes}

for subtype in subtypes :
    for stage in big_dic[subtype].keys() :
        for channel in big_dic[subtype][stage].keys() :
            dic_psd[subtype][stage][channel] = np.nanmean(big_dic[subtype][stage][channel], axis = 0)
            dic_sem[subtype][stage][channel] = sem(big_dic[subtype][stage][channel], nan_policy = 'omit')
                
with open(big_av_psd_savepath, 'wb') as handle:
    pickle.dump(big_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(big_av_sem_savepath, 'wb') as handle:
    pickle.dump(dic_sem, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% 

# palette = ["#5e6472", "#faa307"]

for stage in stages:
    # Create a new figure with three subplots
    fig, axs = plt.subplots(
        nrows=1, ncols=3, figsize=(10, 16), sharey=True, layout = "constrained")

    # Loop through each channel
    for i, channel in enumerate(channels):
        ax = axs[i]

        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes):
            # Convert power to dB
            psd_db = dic_psd[subtype][stage][channel]

            # Calculate the SEM
            sem_db = dic_sem[subtype][stage][channel]

            # Plot the PSD and SEM
            ax.plot(
                freqs, 
                psd_db, 
                label = subtype, 
                color = psd_palette[j],
                alpha = .7,
                linewidth = 2
                )
            ax.fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db, 
                alpha=0.2, 
                color = psd_palette[j]
                )

        # Set the title and labels
        ax.set_title('Channel: ' + channel)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim([0.5, 40])
        # ax.set_ylim([-30, 60])
        ax.legend()

    # Add the condition name as a title for the entire figure
    fig.suptitle('Condition: ' + stage)

    # Add a y-axis label to the first subplot
    axs[0].set_ylabel('Power (dB)')
    axs[0].get_legend().set_visible(False)
    axs[1].get_legend().set_visible(False)

    # Adjust the layout of the subplots
    # plt.constrained_layout()

    # Show the plot
    plt.show()
    # fig_savename = (fig_dir + "/flatPSD_plot_" 
    #                 + stage + ".png")
    # plt.savefig(fig_savename, dpi = 300)

# %% 
from mne.stats import permutation_cluster_test
import scipy
import seaborn as sns

alpha_cluster_forming = 0.05
n_conditions = 2
n_observations = 80
dfn = n_conditions - 1
dfd = n_observations - n_conditions

for stage in stages :
    fig, axs = plt.subplots(
        nrows=1, ncols=3, figsize=(10, 4), sharey=True, layout = "constrained")
    for i_ch, channel in enumerate(channels) :

        hsi_power = np.dstack(
            [i for i in big_dic['HI'][stage][channel]]).transpose((2, 1, 0))
        ctl_power = np.dstack(
            [i for i in big_dic['C1'][stage][channel]]).transpose((2, 1, 0))

        f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)
        
        F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [ctl_power, hsi_power],
            out_type="mask",
            n_permutations=1000,
            threshold=f_thresh,
            tail=0
            )

        # Loop through each population and plot its PSD and SEM
        ax = axs[i_ch]
        for j, subtype in enumerate(subtypes):
            # Convert power to dB
            psd_db = gaussian_filter(dic_psd[subtype][stage][channel], 1)

            # Calculate the SEM
            sem_db = gaussian_filter(dic_sem[subtype][stage][channel], 1)

            # Plot the PSD and SEM
            ax.plot(
                freqs, 
                psd_db, 
                label = subtype, 
                color = psd_palette[j],
                alpha = .7,
                linewidth = 2
                )
            ax.fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db, 
                color = psd_palette[j],
                alpha=0.2, 
                )

        for i_c, c in enumerate(clusters):
            c = np.squeeze(c)
            # if np.any(c) :
            if cluster_p_values[i_c] <= 0.05:
                # h = ax.axvspan(freqs[c].min(), freqs[c].max(), color="r", alpha=0.1)
                ax.plot(
                    [freqs[c].min(), freqs[c].max()], 
                    [ax.get_ylim()[1], ax.get_ylim()[1]], 
                    color="r", 
                    linewidth=3
                    )
            
            # hf = plt.plot(freqs, T_obs, "g")
                # ax.legend((h,), ("cluster p-value < 0.05",))
        sns.despine()
        # Set the title and labels
        # ax.set_title('Channel: ' + channel)
        ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 18)
        ax.set_xlim([0.5, 40])
        # ax.set_ylim([-30, 60])
        # ax.legend()

        # Add the condition name as a title for the entire figure
        # fig.suptitle('Condition: ' + stage)

        # Add a y-axis label to the first subplot
    axs[0].set_ylabel('Power', font = bold_font, fontsize = 18)
    # axs[0].get_legend().set_visible(False)
    # axs[1].get_legend().set_visible(False)
    plt.show(block = False)
    fig_savename = (fig_dir + "/flatPSD_plot_clusterperm" 
                    + stage + ".png")
    plt.savefig(fig_savename, dpi = 300)

