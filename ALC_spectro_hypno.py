#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:43:33 2024

@author: arthurlecoz

ALC_spectro_hypno.py

"""
# %% Paths
import mne, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.stats import sem
from fooof import FOOOFGroup
from fooof.sim.gen import gen_aperiodic

from datetime import date
todaydate = date.today().strftime("%d%m%y")

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = f"{root_dir}/Raw"
preproc_dir = f"{root_dir}/Preproc"
fig_dir = f"{root_dir}/Figs"

epochs_files = glob(os.path.join(
    preproc_dir, "*PSG1*.fif")
    )

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

subtypes = ["C1", "N1"]
channels = ["F3", "C3", "O1"]
psd_palette = ["#1f77b4", "#ff7f0e"]
sem_palette = ['#c9e3f6', '#ffddbf']
freqs = np.arange(0.5, 40.5, 0.5)

# %% Fun

# Generating spectrogram and hypnogram plot
def plot_spectrogram_hypnogram(epochs, freq_min=0.5, freq_max=40):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Computing the Morlet wavelet transform
    tfr = mne.time_frequency.tfr_morlet(
        epochs, 
        freqs=np.linspace(freq_min, freq_max, 100),
        n_cycles=2, 
        return_itc=False)
    
    # Plotting the spectrogram
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}
        )
    tfr.plot(
        picks=[0], axes=axes[0], colorbar=True, show=False, cmap='viridis'
        )
    axes[0].set_title('Spectrogram')
    
    # Plotting the hypnogram
    hypnogram = epochs.metadata['scoring'].values
    axes[1].plot(epochs.times, hypnogram, label='Hypnogram', color='k')
    axes[1].set_yticks(np.unique(hypnogram))
    axes[1].set_yticklabels(np.unique(hypnogram))
    axes[1].set_xlabel('Time (s)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# %% Loop
stages = ["WAKEb", "WAKEd", "N1", "N2", "N3", "REM"]

big_dic = {subtype : {stage : [] for stage in stages} 
            for subtype in subtypes}

for file in epochs_files:
    key = file.split('Preproc/')[-1].split('_')[0]
    if key.startswith('H') : continue
    sub_id = file.split('Preproc/')[-1].split('_PSG1')[0]
    
    print(f"...processing {sub_id}")
    
    epochs = mne.read_epochs(file, preload = True)
    
    metadata = epochs.metadata.reset_index()
    orderedMetadata = metadata.sort_values("n_epoch")
    
    epochsWakeAfterNight = []
    while orderedMetadata.divBin2.iloc[-1] == "noBin" :
        epochsWakeAfterNight.append(orderedMetadata.n_epoch.iloc[-1])
        orderedMetadata = orderedMetadata.iloc[:-1]
        
    for thisEpoch in epochsWakeAfterNight :
        metadata = metadata.loc[metadata['n_epoch'] != thisEpoch]
    epochs = epochs[np.asarray(metadata.index)]
    # epochs = epochs[epochs.metadata.night_status == 'inNight']
    
    bigstagelist = []
    for i_st, stage in enumerate(metadata.scoring):
        if stage == 'WAKE' and metadata.night_status.iloc[i_st] == "outNight" :
            bigstagelist.append('WAKEb')
        elif stage == 'WAKE' and metadata.night_status.iloc[i_st] == "inNight" :
            bigstagelist.append('WAKEd')
        else :
            bigstagelist.append(stage)
    epochs.metadata["newstage"] = bigstagelist
    metadata = epochs.metadata.reset_index()
    
    thisCount = {}
    for stage in metadata.newstage.unique() :
        thisCount[stage] = np.count_nonzero(
            metadata.newstage == stage)/2
    
    for stage in stages :
        # if stage == 'WAKE' and stage not in epochs.metadata.scoring.unique():
        #     continue
        # if stage == 'NREM' :
        #     big_dic[key][stage].append(
        #         np.mean(
        #             (epochs['N2', 'N3'].compute_psd(
        #                 method = "welch",
        #                 fmin = 0.1, 
        #                 fmax = 40,
        #                 n_fft = 512,
        #                 n_overlap = 123,
        #                 n_per_seg = 256,
        #                 window = "hamming",
        #                 picks = channels)
        #             ),
        #             axis = 0)
        #         )
        if stage not in epochs.metadata.newstage.unique() :
            big_dic[key][stage].append(np.nan * np.empty((3, 80)))
        else :
            temp_power = epochs[
                epochs.metadata.newstage == stage].compute_psd(
                    method = "welch",
                    fmin = 0.1, 
                    fmax = 40,
                    n_fft = 512,
                    n_overlap = 256,
                    n_per_seg = 512,
                    window = "hamming",
                    picks = channels
                    )
            # psd = np.mean(temp_power, axis = 0)
            psd = temp_power.get_data()[:, 1]
            fm = FOOOFGroup(peak_width_limits = [2, 8], aperiodic_mode="fixed")
            fm.add_data(temp_power.freqs, psd)
            fm.fit()

dic_psd = {"C1" : {}, "HI" : {}}
dic_sem = {"C1" : {}, "HI" : {}}

for subtype in subtypes :
    for stage in stages :
        dic_psd[subtype][stage] = 10 * np.log10(np.nanmean(
                big_dic[subtype][stage], axis = 0))
        dic_sem[subtype][stage] = sem(10 * np.log10(
                big_dic[subtype][stage]), axis = 0, nan_policy = "omit")


# %% Plots PSD

for stage in stages:
    # Create a new figure with three subplots
    fig, axs = plt.subplots(
        nrows=1, ncols=3, figsize=(16, 12), sharey=True, layout = "constrained")

    # Loop through each channel
    for i, channel in enumerate(channels):
        ax = axs[i]

        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes):
            # Convert power to dB
            psd_db = dic_psd[subtype][stage][i]

            # Calculate the SEM
            sem_db = dic_sem[subtype][stage][i]

            # Plot the PSD and SEM
            ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
            ax.fill_between(
                freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
                color = sem_palette[j]
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

    # Adjust the layout of the subplots
    # plt.constrained_layout()

    # Show the plot
    plt.show()
    fig_savename = (fig_dir + "/PSD_plot_" 
                    + stage + ".png")
    plt.savefig(fig_savename, dpi = 300)
    
# %% Plot PSD Smoothed

smooth = 1 

for stage in stages:
    # Create a new figure with three subplots
    fig, axs = plt.subplots(
        nrows=1, ncols=3, figsize=(16, 12), sharey=True, layout = "constrained")

    # Loop through each channel
    for i, channel in enumerate(channels):
        ax = axs[i]

        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes):
            # Convert power to dB
            psd_db = gaussian_filter(dic_psd[subtype][stage][i], smooth)

            # Calculate the SEM
            sem_db = gaussian_filter(dic_sem[subtype][stage][i], smooth)

            # Plot the PSD and SEM
            ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
            ax.fill_between(
                freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
                color = sem_palette[j]
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

    # Adjust the layout of the subplots
    # plt.constrained_layout()

    # Show the plot
    plt.show()
    fig_savename = (fig_dir + "/smoothed_PSD_plot_" 
                    + stage + ".png")
    plt.savefig(fig_savename, dpi = 300)

# %% perm clust 
# Could be better (
    # -> one loop across stages & p value of each cluster

from mne.stats import permutation_cluster_test
import scipy

hsi_power = np.dstack([i for i in big_dic['HI']['N2']]).transpose((2, 1, 0))
ctl_power = np.dstack([i for i in big_dic['C1']['N2']]).transpose((2, 1, 0))

adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, 'eeg')

alpha_cluster_forming = 0.05
n_conditions = 2
n_observations = 80
dfn = n_conditions - 1
dfd = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    [ctl_power, hsi_power],
    out_type="mask",
    n_permutations=1000,
    threshold=f_thresh,
    tail=0,
    adjacency = adjacency
    )

# %% plot w/ stats

fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize=(6, 12), sharey=True, layout = "constrained")

# Loop through each channel
# for i, channel in enumerate(channels):
#     ax = axs[i]

i = 1
channel = 'O1'
stage = 'N2'

# Loop through each population and plot its PSD and SEM
for j, subtype in enumerate(subtypes):
    # Convert power to dB
    psd_db = gaussian_filter(dic_psd[subtype][stage][i], 1)

    # Calculate the SEM
    sem_db = gaussian_filter(dic_sem[subtype][stage][i], 1)

    # Plot the PSD and SEM
    ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
    ax.fill_between(
        freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
        color = sem_palette[j]
        )

for i_c, c in enumerate(clusters):
    c = c[:,2]
    if np.any(c) :
    # if cluster_p_values[i_c] <= 0.05:
        h = ax.axvspan(freqs[c].min(), freqs[c].max(), color="r", alpha=0.1)
    
    
    # hf = plt.plot(freqs, T_obs, "g")
        ax.legend((h,), ("cluster p-value < 0.05",))

# Set the title and labels
ax.set_title('Channel: ' + channel)
ax.set_xlabel('Frequency (Hz)')
ax.set_xlim([0.5, 40])
# ax.set_ylim([-30, 60])
ax.legend()

# Add the condition name as a title for the entire figure
fig.suptitle('Condition: ' + stage)

# Add a y-axis label to the first subplot
ax.set_ylabel('dB')
plt.show(block = False)
       
# %% plot FLAT PSD 

n_chs = 3

for i_stage, stage in enumerate(stages) :
    fig, ax = plt.subplots(
        nrows = 1, ncols = 3, sharex = True, sharey = True, layout = "tight",
        figsize = (12, 12)
        )
    for i_ch, chan in enumerate(["F3", "C3", "O1"]) :
        for i_subtype, subtype in enumerate(["C1", "HI"]) :

            this_psd = dic_psd[subtype][stage][i]
            
            unlogged_psd = 10**(this_psd/10)
            
            fm = FOOOF(peak_width_limits = [1, 4], aperiodic_mode = 'fixed')
            fm.fit(freqs, unlogged_psd)
            init_ap_fit = gen_aperiodic(
                fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
                )    
            flattened_spectra = fm.power_spectrum - init_ap_fit
            del fm
            
            ax[i_ch].plot(
                freqs, flattened_spectra, color = psd_palette[i_subtype], 
                # ls = ls_type[epoch]
                )
            ax[i_ch].set_title(f"{chan}")
            ax[0].set_xlabel("Frequency (Hz)")
            ax[0].set_ylabel("Log Power (dB / Hz)")
            # ax[0].set_ylim(-1, 1)
            fig.suptitle(
                f"Aperiodic PSD comparison during {stage}"
                )
    plt.savefig(f"{fig_dir}/{stage}_FOOOF_flattened_PSD.png", dpi = 200)
       