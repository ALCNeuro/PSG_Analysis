#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:15:17 2023

@author: arthurlecoz

ALC_explore_fooof_peaks.py

4s windows 
smooth the signal over 2 Hz
lowess smoothing is better than gaussian filter > Follows the signal 
=> Does not try and fit an actual signal

Lowess smooth before fooof


"""
# %% Paths
import mne, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.stats import sem
from fooof import FOOOFGroup
from fooof.plts.periodic import plot_peak_fits, plot_peak_params
from fooof.analysis import get_band_peak_fg, get_band_peak_fm
from fooof.objs import combine_fooofs, average_fg
from fooof.bands import Bands
import pickle

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

subtypes = ["C1", "N1", "HI"]
channels = ["F3", "C3", "O1"]
psd_palette = ["#1f77b4", "#ff7f0e"]
sem_palette = ['#c9e3f6', '#ffddbf']

features = [
    "sub_id", "subtype", "age", "genre", "stage", "channel", 
    "duration_wakeb","duration_waked", "duration_N1", "duration_N2","duration_N3", "duration_REM", 
    "exponent", "offset"
    ]

# %% Loop

bigdic_savepath = os.path.join(
    fig_dir, "NT1_CnS_FOOOF"
    )

dic_fooof = {feat : [] for feat in features}

stages = ["WAKEb", "WAKEd", "N1", "N2", "N3", "REM"]
freqs = np.arange(0.5, 40.5, 0.5)

bands = Bands({
    "delta" : (.5, 4),
    "theta" : (4, 8),
    "alpha" : (8, 12),
    "sigma" : (12, 16),
    "iota" : (25, 35)
    })

big_dic = {subtype : {stage : {channel: [] for channel in channels}
                      for stage in stages} 
            for subtype in subtypes}

for file in epochs_files:
    key = file.split('Preproc/')[-1].split('_')[0]
    # if key.startswith('H') : continue
    sub_id = file.split('Preproc/')[-1].split('_PSG1')[0]
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
        if stage not in epochs.metadata.newstage.unique() : continue
        else : 
            temp_power = epochs[
                epochs.metadata.newstage == stage].compute_psd(
                    method = "welch",
                    fmin = 0.5, 
                    fmax = 40,
                    n_fft = 512,
                    n_overlap = 256,
                    n_per_seg = 512,
                    window = "hamming",
                    picks = channels
                    )
            for i_ch, channel in enumerate(channels) :
                psd = temp_power.get_data()[:, i_ch]
                fg = FOOOFGroup(peak_width_limits = [2, 8], aperiodic_mode="fixed")
                fg.add_data(freqs, psd)
                fg.fit()
                fm = average_fg(fg, bands)
                offset, exponent = fm.aperiodic_params_
                big_dic[key][stage][channel].append(fm)
                
                dic_fooof["sub_id"].append(sub_id)
                dic_fooof["subtype"].append(key)
                dic_fooof["age"].append(age)
                dic_fooof["genre"].append(genre)
                dic_fooof["stage"].append(stage)
                dic_fooof["channel"].append(channel)
                
                if np.count_nonzero(metadata.newstage == "WAKEb") == 0 : 
                    dic_fooof["duration_wakeb"].append(0)
                else :
                    dic_fooof["duration_wakeb"].append(thisCount["WAKEb"])
                if np.count_nonzero(metadata.newstage == "WAKEd") == 0 : 
                    dic_fooof["duration_waked"].append(0)
                else :
                    dic_fooof["duration_waked"].append(thisCount["WAKEd"])
                if np.count_nonzero(metadata.newstage == "N1") == 0 :
                    dic_fooof["duration_N1"].append(0)
                else :
                    dic_fooof["duration_N1"].append(thisCount["N1"])
                dic_fooof["duration_N2"].append(thisCount["N2"])
                dic_fooof["duration_N3"].append(thisCount["N3"])
                dic_fooof["duration_REM"].append(thisCount["REM"])
                
                dic_fooof['offset'].append(offset)
                dic_fooof['exponent'].append(exponent)
                
df = pd.DataFrame.from_dict(dic_fooof)
df.to_csv(os.path.join(
    fig_dir, "fooof", "new_expo_offset_allsubtype.csv"
    ))
with open(bigdic_savepath, 'wb') as f:
    pickle.dump(big_dic, f)
            
# %% 

with open(bigdic_savepath, 'rb') as handle:
    big_dic = pickle.load(handle)

# %% 

colors = ['#2400a8', '#00700b']
labels = ['C1', 'N1']
bands = Bands({
    "delta" : (.5, 4),
    "theta" : (4, 8),
    "alpha" : (8, 12),
    "sigma" : (12, 16),
    "iota" : (25, 35)
    })

for stage in ['WAKEd', 'WAKEb', 'REM'] : 
    for channel in channels : 
        fg_C1 = combine_fooofs(big_dic['C1'][stage][channel])
        fg_N1 = combine_fooofs(big_dic['N1'][stage][channel])
        peak_C1 = get_band_peak_fg(fg_C1, (.5, 40))
        peak_N1 = get_band_peak_fg(fg_N1, (.5, 40))
        plot_peak_params([peak_C1, peak_N1] ,labels = labels, colors = colors)
        plt.title(f"{stage} - {channel}")
        plot_peak_fits([peak_C1, peak_N1], labels = labels, colors = colors)
        plt.title(f"{stage} - {channel}")
