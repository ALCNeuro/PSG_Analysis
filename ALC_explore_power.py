#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:15:17 2023

@author: arthurlecoz

PLOT_psd_night.py

---
01/05 : Concatenate epochs do not work to compute psd
    -> Could try evoked :
        - create a dic for every stage
        - dic would be appending evoked for each subject
        - i could have a dic inside a dic w/ every subtypes
    
    Or I do a quick script to 

"""
# %% Paths
import mne, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.stats import sem
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from mne.stats import permutation_cluster_test
import scipy
import pickle
from statsmodels.nonparametric.smoothers_lowess import lowess

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from datetime import date
todaydate = date.today().strftime("%d%m%y")

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

which = "all" # "nt1_cns" or "all"

if which == "nt1_cns" :
    subtypes = ["C1", "N1"]
    palette = ["#8d99ae", "#d00000"]
    psd_palette = ["#8d99ae", "#d00000"]
    sem_palette = ['#D7DBE2', '#fca5a5']
elif which == "all" :
    subtypes = ["C1", "N1", 'HI']
    palette = ["#8d99ae", "#d00000", "#ffb703"]
    psd_palette = ["#8d99ae", "#d00000", "#ffb703"]
    # sem_palette = ['#D7DBE2', '#fca5a5']
    
channels = ["F3", "C3", "O1"]
freqs = np.linspace(0.5, 40, 159)

# %% Loop

big_dic_psd_savepath = os.path.join(
    fig_dir, "psd_dic.pickle"
    )

stages = ["WAKEb", "WAKEd", "N1", "N2", "N3", "REM"]

if os.path.exists(big_dic_psd_savepath) :
    big_dic = pd.read_pickle(big_dic_psd_savepath)

else : 
    big_dic = {subtype : {stage : {chan : [] for chan in channels}
                          for stage in stages} for subtype in subtypes}
    
    for file in epochs_files :
        key = file.split('/')[-1].split('_')[0]
        sub_id = file.split('/')[-1].split('_PSG1')[0]
        # if sub_id.startswith("H") :
        #     continue
        
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
            if stage not in epochs.metadata.newstage.unique() : 
                for i_ch, channel in enumerate(channels) :
                    big_dic[key][stage][channel].append(np.nan*np.empty([len(freqs)]))
            else :
                temp_list = []
                temp_power = epochs[epochs.metadata.newstage == stage].compute_psd(
                    method = "welch",
                    fmin = .5, 
                    fmax = 40,
                    n_fft = 1024,
                    n_overlap = 512,
                    n_per_seg = 1024,
                    window = "hamming",
                    picks = channels
                    )
                
                for i_ch, channel in enumerate(channels) :
                    for i_epoch in range(len(epochs[epochs.metadata.newstage == stage])) :
                        this_power = temp_power[i_epoch]
                        psd = lowess(np.squeeze(this_power.copy().pick(channel).get_data()), freqs, 0.075)[:, 1]
                        temp_list.append(psd)
                        
                    big_dic[key][stage][channel].append(np.nanmean(temp_list, axis = 0))
    
    with open(big_dic_psd_savepath, 'wb') as handle:
        pickle.dump(big_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%

dic_psd = {subtype : {stage : {chan : [] for chan in channels}
                      for stage in stages} for subtype in subtypes}
dic_sem = {subtype : {stage : {chan : [] for chan in channels}
                      for stage in stages} for subtype in subtypes}

for subtype in subtypes :
    for stage in stages :
        for channel in channels :
            dic_psd[subtype][stage][channel] = 10 * np.log10(np.nanmean(
                    big_dic[subtype][stage][channel], axis = 0))
            dic_sem[subtype][stage][channel] = sem(10 * np.log10(
                    big_dic[subtype][stage][channel]), axis = 0, nan_policy = "omit")

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
            psd_db = dic_psd[subtype][stage][channel]

            # Calculate the SEM
            sem_db = dic_sem[subtype][stage][channel]

            # Plot the PSD and SEM
            ax.plot(freqs, psd_db, label = subtype, color = palette[j])
            ax.fill_between(
                freqs, psd_db - sem_db, psd_db + sem_db, alpha=0.3, 
                color = palette[j]
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
    fig_savename = (fig_dir + "/PSD_nt1_hi_cns_plot_" 
                    + stage + ".png")
    plt.savefig(fig_savename, dpi = 300)
    
# %% plot subjects and mean

dic_st = {
    "C1" : "Healthy Sleepers",
    "N1" : "Narcolepsy type 1 Patients",
    "HI" : "Idiopathic Hypersomnia Long Sleep Time"
    }

for stage in stages:
    # Loop through each population and plot subject PSD
    for j, subtype in enumerate(subtypes):
        # Create a new figure with three subplots
        fig, axs = plt.subplots(
            nrows=1, ncols=3, figsize=(16, 12), sharey=True, layout = "constrained")
        # Loop through each channel
        for i, channel in enumerate(channels):
            ax = axs[i]
            for s in range(len(big_dic[subtype][stage][channel])) :
        
                # Convert power to dB
                psd_sub = 10 * np.log10(big_dic[subtype][stage][channel][s])
    
                # Plot the PSD and SEM
                ax.plot(
                    freqs, 
                    psd_sub, 
                    color = "lightgrey",
                    alpha = .5
                    )
            
            psd_group = dic_psd[subtype][stage][channel]
            ax.plot(
                freqs, 
                psd_group, 
                color = "red",
                alpha = .8
                )

            # Set the title and labels
            ax.set_title('Channel: ' + channel)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xlim([0.5, 40])
            # ax.set_ylim([-30, 60])

        # Add the condition name as a title for the entire figure
        fig.suptitle(f'Condition: {dic_st[subtype]} during {stage}')

        # Add a y-axis label to the first subplot
        axs[0].set_ylabel('Power (dB)')

    # Adjust the layout of the subplots
    # plt.constrained_layout()

        # Show the plot
        plt.show()
        plt.savefig(os.path.join(fig_dir, f"{subtype}_{stage}_mean_indivsub.png"), 
                    dpi = 200)
    
# %% MNE Report, all subjects PSDs per stage

sub_ids = [file.split('/')[-1].split('_PSG1')[0] for file in epochs_files]

n1_subjects = [subid for subid in sub_ids if subid.startswith("N")]
hi_subjects = [subid for subid in sub_ids if subid.startswith("H")]
c1_subjects = [subid for subid in sub_ids if subid.startswith("C")]

for subtype in subtypes :
    report = mne.Report(title=f"PSDs by subjects for {dic_st[subtype]}")
    
    if subtype == "N1" :
        subjects = n1_subjects.copy()
    elif subtype == "C1" :
        subjects = c1_subjects.copy()
    elif subtype == "HI" :
        subjects = hi_subjects.copy()
    
    for s in range(len(big_dic[subtype][stage][channel])) :
        fig, axs = plt.subplots(
            nrows=1, 
            ncols=3, 
            figsize=(10, 6), 
            sharey=True, 
            layout = "constrained"
            )
        for stage in stages :
            if stage == 'WAKEb':continue
            for i, channel in enumerate(channels) :
                ax = axs[i]
                psd_sub = 10 * np.log10(big_dic[subtype][stage][channel][s])
                ax.plot(
                    freqs, 
                    psd_sub, 
                    # color = "lightgrey",
                    alpha = .8,
                    label = stage
                    )
                ax.set_title('Channel: ' + channel)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_xlim([0.5, 40])
                # ax.set_ylim([-30, 60])

            # Add the condition name as a title for the entire figure
            fig.suptitle(f'PSDs of {subjects[s]}')

            # Add a y-axis label to the first subplot
            axs[0].set_ylabel('Power (dB)')
            axs[2].legend()
        report.add_figure(
            fig=fig,
            title=f"PSD of {subjects[s]} during {stage}",
            image_format="PNG",
            tags=(subtype, stage)
            )
        plt.close()
    report.save(os.path.join(
        fig_dir, f"{subtype}_psds.html"), 
        open_browser = False,
        overwrite=True
        )

    
# %% Plot PSD Smoothed

# smooth = 2
# subtypes = ['C1', 'N1']

# for stage in stages:
#     # Create a new figure with three subplots
#     fig, axs = plt.subplots(
#         nrows=1, ncols=3, figsize=(16, 12), sharey=True, layout = "constrained")

#     # Loop through each channel
#     for i, channel in enumerate(channels):
#         ax = axs[i]

#         # Loop through each population and plot its PSD and SEM
#         for j, subtype in enumerate(subtypes):
#             # Convert power to dB
#             # psd_db = gaussian_filter(dic_psd[subtype][stage][i], smooth)
#             psd_db = lowess(dic_psd[subtype][stage][channel], freqs, .1)[:, 1]

#             # Calculate the SEM
#             # sem_db = gaussian_filter(dic_sem[subtype][stage][i], smooth)
#             sem_db = lowess(dic_sem[subtype][stage][channel], freqs, .1)[:, 1]

#             # Plot the PSD and SEM
#             ax.plot(freqs, psd_db, label = subtype, color = palette[j], alpha=0.9)
#             ax.plot(freqs, psd_db, label = subtype, color = palette[j], alpha=0.9)
#             ax.fill_between(
#                 freqs, psd_db - sem_db, psd_db + sem_db, alpha=0.25, 
#                 color = palette[j]
#                 )

#         # Set the title and labels
#         ax.set_title('Channel: ' + channel)
#         ax.set_xlabel('Frequency (Hz)')
#         ax.set_xlim([0.5, 40])
#         # ax.set_ylim([-30, 60])
#         ax.legend()

#     # Add the condition name as a title for the entire figure
#     fig.suptitle('Condition: ' + stage)

#     # Add a y-axis label to the first subplot
#     axs[0].set_ylabel('Power (dB)')

#     # Adjust the layout of the subplots
#     # plt.constrained_layout()

#     # Show the plot
#     plt.show()
    # fig_savename = (fig_dir + "/smoothed_PSD_nt1_hi_cns_plot_" 
    #                 + stage + ".png")
    # plt.savefig(fig_savename, dpi = 300)

# %% perm clust 
# Could be better (
    # -> one loop across stages & p value of each cluster

hsi_power = np.dstack([i for i in big_dic['C1']['REM']['O1']]).transpose((2, 1, 0))
ctl_power = np.dstack([i for i in big_dic['N1']['REM']['O1']]).transpose((2, 1, 0))

# hsi_power = np.dstack([i for i in big_dic['N1']['N2']]).transpose((2, 1, 0))
# ctl_power = np.dstack([i for i in big_dic['HI']['N2']]).transpose((2, 1, 0))

# adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, 'eeg')

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
    # adjacency = adjacency
    )

# %% plot w/ stats

chan_names = ["F3", "C3", "O1"]
subtypes_here = ["C1", "N1"]

# adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, 'eeg')

for i_st, stage in enumerate(stages) : 
    
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 12), sharey=True, layout = "constrained")
    for i_ch, channel in enumerate(chan_names) :
        
        hsi_power = np.dstack(
            [i for i in big_dic[subtypes_here[0]][stage][channel]]
            ).transpose((2, 1, 0))
        ctl_power = np.dstack(
            [i for i in big_dic[subtypes_here[1]][stage][channel]]
            ).transpose((2, 1, 0))

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
            # threshold=f_thresh,
            # tail=0,
            # adjacency = adjacency
            )
        
        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes_here):
            # Convert power to dB
            psd_db = dic_psd[subtype][stage][channel]
        
            # Calculate the SEM
            sem_db = dic_sem[subtype][stage][channel]
        
            # Plot the PSD and SEM
            ax[i_ch].plot(
                freqs, 
                psd_db, 
                label = subtype, 
                color = psd_palette[j]
                )
            ax[i_ch].fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db, 
                alpha=0.3, 
                color = psd_palette[j]
                )
        
        for i_c, c in enumerate(clusters):
            # c = c[:, i_ch]
            c = np.squeeze(c)
            if np.any(c) :
                if cluster_p_values[i_c] <= 0.05:
                    h = ax[i_ch].axvspan(freqs[c].min(), freqs[c].max(), color="r", alpha=0.1)
            
            
            # hf = plt.plot(freqs, T_obs, "g")
                # ax[i_ch].legend((h,), ("cluster p-value < 0.05",))
        
        # Set the title and labels
        ax[i_ch].set_title('Channel: ' + chan_names[i_ch])
        ax[i_ch].set_xlabel('Frequency (Hz)')
        ax[i_ch].set_xlim([0.5, 40])
        # ax.set_ylim([-30, 60])
        ax[i_ch].legend()
        
        # Add the condition name as a title for the entire figure
        fig.suptitle('Condition: ' + stage)
        
        # Add a y-axis label to the first subplot
        ax[i_ch].set_ylabel('dB')
    plt.show(block = False)
    plt.savefig(os.path.join(
        fig_dir, f"nt1vcns_{stage}_psd_cluster.png"
        ), dpi = 300)

# %% Bandpower

freqs = np.linspace(0.5, 40, 80)
bands = {
    "delta" : (1,  4),
    "theta" : (4 ,  8),
    "alpha" : (8 , 12),
    "sigma"  : (12, 16),
    "beta"  : (16, 30)
    }

basic_params = [
    "sub_id", "subtype", "stage", "channel", "age", "genre",
    "dur_WAKEb", "dur_WAKEd", "dur_N1", "dur_N2", "dur_N3", "dur_REM",
    "abs_delta", "abs_theta", "abs_alpha", "abs_sigma", "abs_beta",
    "rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "rel_beta",
    ]

big_dic = {param : [] for param in basic_params}

thisSavingPath = os.path.join(
    fig_dir, "bandpower_nt1_cns_df.csv"
    )

if os.path.exists(thisSavingPath) :
    df = pd.read_csv(thisSavingPath)
    del df['Unnamed: 0']
else : 
    for file in glob(f"{preproc_dir}/*PSG1*.fif"):
        key = file.split('Preproc/')[-1].split('_')[0]
        sub_id = file.split('Preproc/')[-1].split('_PSG1')[0]
        
        if sub_id.startswith("H") :
            continue
        thisDemo = df_demographics.loc[df_demographics.code == sub_id]
        age = thisDemo.age.iloc[0]
        genre = thisDemo.sexe.iloc[0]
        
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
            if stage not in epochs.metadata.newstage.unique() :
                continue
            else : 
                temp_power = np.mean(
                        (epochs[epochs.metadata.newstage == stage].compute_psd(
                            method = "welch",
                            fmin = .5, 
                            fmax = 40,
                            n_fft = 512,
                            n_overlap = 128,
                            n_per_seg = 256,
                            window = "hamming",
                            n_jobs = 4)
                        ),
                        axis = 0)
    
            for i_ch, chan in enumerate(epochs.ch_names) : 
                thischan_power = temp_power[i_ch, :]
                thistot_power = sum([
                    np.mean(thischan_power[
                        np.logical_and(freqs >= borders[0], freqs <= borders[1])
                        ], axis = 0) for band, borders in bands.items()
                    ])
                big_dic["sub_id"].append(sub_id)
                big_dic["age"].append(age)
                big_dic["genre"].append(genre)
                big_dic["subtype"].append(key)
                big_dic["channel"].append(chan)
                big_dic["stage"].append(stage)
                if np.count_nonzero(metadata.newstage == "WAKEb") == 0 : 
                    big_dic["dur_WAKEb"].append(np.nan)
                else :
                    big_dic["dur_WAKEb"].append(thisCount["WAKEb"])
                if np.count_nonzero(metadata.newstage == "WAKEd") == 0 : 
                    big_dic["dur_WAKEd"].append(np.nan)
                else :
                    big_dic["dur_WAKEd"].append(thisCount["WAKEd"])
                if np.count_nonzero(metadata.newstage == "N1") == 0 :
                    big_dic["dur_N1"].append(np.nan)
                else :
                    big_dic["dur_N1"].append(thisCount["N1"])
                big_dic["dur_N2"].append(thisCount["N2"])
                big_dic["dur_N3"].append(thisCount["N3"])
                big_dic["dur_REM"].append(thisCount["REM"])
                for band, borders in bands.items() :
                    abs_bandpower = np.mean(thischan_power[
                        np.logical_and(freqs >= borders[0], freqs <= borders[1])
                        ], axis = 0)
                    rel_bandpower = abs_bandpower/thistot_power
                        
                    big_dic[f"abs_{band}"].append(10 * np.log(abs_bandpower))
                    big_dic[f"rel_{band}"].append(rel_bandpower)
         
    df = pd.DataFrame.from_dict(big_dic)
    # for col in df.columns[-4:] :
    #     df[col] = 10 * np.log10(df[col])
    df.to_csv(thisSavingPath)
          
# %%

import seaborn as sns

feature = "abs_beta"
features = [
    'abs_delta','abs_theta', 'abs_alpha', 'abs_sigma', 'abs_beta', 
    'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta'
    ]

for feature in features :
    fig, ax = plt.subplots(
        nrows = 3, ncols = 1, sharex = True, sharey = True, figsize = (6, 16))
    
    for i_ch, channel in enumerate(df.channel.unique()) :
        sns.violinplot(
            data = df,
            x = "stage", 
            order = df.stage.unique(),
            y = feature,
            inner = 'quartile',
            ax = ax[i_ch],
            hue = "subtype",
            palette = palette,
            cut = 2
            # hue_order = ["CTL", "NT1"]
            )
        ax[i_ch].set_title(channel)
    
    fig.tight_layout(pad = .5)
    plt.savefig(os.path.join(
        fig_dir, f"nt1vcns_{feature}_violin.png"), 
        dpi = 300
        )

# %% LME 

this_df = df.loc[df.stage != 'WAKEb']

if which == "nt1_cns" :

    foi = ["Stage", "Channel", "ß (NT1 vs HS)", "p_val"]
    dic = {f : [] for f in foi}
    
    for stage in this_df.stage.unique():
        for channel in df.channel.unique() :
            model_formula = f'rel_alpha ~ age + C(genre) + C(subtype, Treatment("C1")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
            model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing='drop')
            model_result = model.fit()
            
            dic["Stage"].append(stage)
            dic["Channel"].append(channel)
            dic["ß (NT1 vs HS)"].append(
                model_result.params['C(subtype, Treatment("C1"))[T.N1]'])
            dic["p_val"].append(
                model_result.pvalues['C(subtype, Treatment("C1"))[T.N1]'])
            
            
    corr_pval = multipletests(dic["p_val"], method='fdr_tsbh')
    print(np.asarray(dic["Channel"])[corr_pval[0]])
    print(np.asarray(dic["Stage"])[corr_pval[0]])
    print(np.asarray(corr_pval[1])[corr_pval[0]])
    dic['p_corr'] = list(corr_pval[1])
    stats_df = pd.DataFrame.from_dict(dic)
    print(stats_df)
    
elif which == "all" :
    
    foi = ["Stage", "Channel", "ß (NT1 vs HS)", "ß (IH_LTS vs HS)", "ß (IH_LTS vs NT1)", "p_val"]
    dic = {f : [] for f in foi}
    
    for stage in this_df.stage.unique():
        for channel in df.channel.unique() :
            model_formula = f'rel_alpha ~ age + C(genre) + C(subtype, Treatment("C1")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
            model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing='drop')
            model_result = model.fit()
            
            dic["Stage"].append(stage)
            dic["Channel"].append(channel)
            dic["ß (NT1 vs HS)"].append(
                model_result.params['C(subtype, Treatment("C1"))[T.N1]'])
            dic["p_val"].append(
                model_result.pvalues['C(subtype, Treatment("C1"))[T.N1]'])
            
            
    corr_pval = multipletests(dic["p_val"], method='fdr_tsbh')
    print(np.asarray(dic["Channel"])[corr_pval[0]])
    print(np.asarray(dic["Stage"])[corr_pval[0]])
    print(np.asarray(corr_pval[1])[corr_pval[0]])
    dic['p_corr'] = list(corr_pval[1])
    stats_df = pd.DataFrame.from_dict(dic)
    print(stats_df)